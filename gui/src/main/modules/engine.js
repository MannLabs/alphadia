const { exec, spawn } = require('child_process');
const StringDecoder = require('string_decoder').StringDecoder;
const Transform = require('stream').Transform;

const { dialog } = require('electron');
const writeYamlFile = require('write-yaml-file')

const { workflowToConfig } = require('./workflows');
const os = require('os');
const fs = require('fs');
const path = require('path');
var kill = require('tree-kill');

function lineBreakTransform () {

    // https://stackoverflow.com/questions/40781713/getting-chunks-by-newline-in-node-js-data-stream
    const decoder = new StringDecoder('utf8');

    return new Transform({
        transform(chunk, encoding, cb) {
        if ( this._last === undefined ) { this._last = "" }
        this._last += decoder.write(chunk);
        var list = this._last.split(/\n/);          
        this._last = list.pop();
        for (var i = 0; i < list.length; i++) {
            this.push( list[i].slice(0, 1000) );
        }
        cb();
    },
    
    flush(cb) {
        this._last += decoder.end()
        if (this._last) { this.push(this._last.slice(0, 1000)) }
        cb()
    }
    });
}

class BaseExecutionEngine {

    name = ""
    valid_plattforms = []
    available = false
    errorMessage = ""
    description = ""
    versions = []

    config = {}

    constructor(config){
        this.config = {...this.config, ...config}
    }

    checkAvailability(){
        // raise error if not implemented by subclass
        throw new Error("checkAvailability() not implemented by subclass")
    }
    
    getStatus(){
        return {
            name: this.name,
            available: this.available,
            error: this.errorMessage,
            description: this.description,
            versions: this.versions
        }
    }
}

function testCommand(command, pathUpdate){
    const PATH = process.env.PATH + ":" + pathUpdate
    return new Promise((resolve, reject) => {
        exec(command, {env: {...process.env, PATH}}, () => {}).on('exit', (code) => {resolve(code)});
    });
}

function buildCondaPATH(username, platform){
    if (platform == "darwin"){
        return [
            "/Users/" + username + "/miniconda3/bin/",
            "/Users/" + username + "/anaconda3/bin/",
            "/Users/" + username + "/miniconda/bin/",
            "/Users/" + username + "/anaconda/bin/",
            "/anaconda/bin/", 
        ]
    } else if (platform == "win32"){
        return [
            "C:\\Users\\" + username + "\\miniconda3\\Scripts\\",
            "C:\\Users\\" + username + "\\anaconda3\\Scripts\\",
            "C:\\Users\\" + username + "\\miniconda\\Scripts\\",
            "C:\\Users\\" + username + "\\anaconda\\Scripts\\",
            "C:\\Users\\" + username + "\\AppData\\Local\\miniconda3\\Scripts\\",
            "C:\\Users\\" + username + "\\AppData\\Local\\anaconda3\\Scripts\\",
            "C:\\Users\\" + username + "\\AppData\\Local\\miniconda\\Scripts\\",
            "C:\\Users\\" + username + "\\AppData\\Local\\anaconda\\Scripts\\"
        ]
    } else {
        return [
            "/opt/conda/bin/",
            "/usr/local/bin/",
            "/usr/local/anaconda/bin/",
        ]
    }
}

function hasConda(profileCondaPath){
    return new Promise((resolve, reject) => {

        const paths = [profileCondaPath, ...buildCondaPATH(os.userInfo().username, os.platform())]

        Promise.all(paths.map((path) => {
            return testCommand("conda", path)
            })).then((codes) => {
                const index = codes.findIndex((code) => code == 0)

                if (index == -1){
                    reject("Conda not found. Please set conda path manually in the settings.")
                } else {
                    resolve(paths[index])
                }
        })
    })
}

function execPATH(command, pathUpdate, callback){
    const PATH = process.env.PATH + ":" + pathUpdate
    exec(command, {env: {...process.env, PATH}}, callback);
}

function condaVersion(condaPath){
    console.log(condaPath)
    return new Promise((resolve, reject) => {
        execPATH("conda info --json", condaPath, (err, stdout, stderr) => {
            if (err) {console.log(err); reject(err); return;}
            const info = JSON.parse(stdout);
            resolve(info["conda_version"]);
        });
    })
}

function hasPython(envName, condaPath){
    return new Promise((resolve, reject) => {
        try {
            execPATH("conda run -n " + envName + " python -c \"print(__import__('sys').version)\"" , condaPath, (err, stdout, stderr) => {
                if (err) {console.log(err); reject("Conda environment " + envName + " not found within WSL"); return;}
                resolve(stdout.trim().split(" ")[0])
            })
        } catch (err) {
            console.log(err)
            reject(err)
        }
    })
}

function hasAlphaDIA(envName, condaPath){
    return new Promise((resolve, reject) => {
        try {
            execPATH(`conda run -n ${envName} alphadia --version` , condaPath, (err, stdout, stderr) => {
                if (err) {console.log(err); reject("alphaDIA not found within WSL"); return;}
                resolve(stdout.trim())
            }
        )} catch (err) {
            console.log(err)
            reject(err)
        }
    })
}
class CMDExecutionEngine extends BaseExecutionEngine {

    name = "CMDExecutionEngine"
    description = "The command line interface (CMD) is being used to run alphaDIA. Recommended execution engine for Linux and MacOS users."
    valid_plattforms = [
        "win32",
        "linux",
        "darwin"
    ]

    discoveredCondaPath = ""

    checkAvailability(){
        return new Promise((resolve, reject) => {
            
            if (this.valid_plattforms.indexOf(process.platform) == -1){
                this.available = false
                this.error = "Plattform " + process.platform + " is not supported when using " + this.constructor.name
                resolve(this)
            }
            
            console.log(this.config)
            return hasConda(this.config.condaPath).then((conda_path) => {
                this.discoveredCondaPath = conda_path
            }).then(() => {
                return condaVersion(this.discoveredCondaPath)
            }).then((conda_version) => {
                this.versions.push({"name": "conda", "version": conda_version})
            }).then(() => {
                return hasPython(this.config.envName, this.discoveredCondaPath)
            }).then((python_version) => {
                this.versions.push({"name": "python", "version": python_version})
            }).then(() => {
                return hasAlphaDIA(this.config.envName, this.discoveredCondaPath)
            }).then((alphadia_version) => {
                this.versions.push({"name": "alphadia", "version": alphadia_version})
            }).then(() => {
                this.available = true
            }).catch((err) => {
                this.available = false
                this.error = err
            }).finally(() => {
                console.log(this.constructor.name + " status")
                console.log(this)
                resolve(this)
            })
        })
    }

    saveWorkflow(workflow){
        return new Promise((resolve, reject) => {
            console.log(workflow)
            const workflowFolder = workflow.output_directory.path
            const config = workflowToConfig(workflow)
            if (!fs.existsSync(workflowFolder)) {
                reject("Output folder " + workflowFolder + " does not exist.")
            }
            const configPath = path.join(workflowFolder, "config.yaml")
            // save config.yaml in workflow folder
            writeYamlFile.sync(configPath, config, {lineWidth: -1})
            resolve()
        })
    }

    startWorkflow(workflow){
        return this.saveWorkflow(workflow).then(() => {

            let run = {
                engine: this.constructor.name,
                name: workflow.name,
                path: workflow.output_directory.path,
                std: [],
                pid: null,
                code: -1,
                process : null,
                activePromise: null,
            }

            const PATH = process.env.PATH + ":" + this.discoveredCondaPath

            run.process = spawn("conda", ["run", 
                                        "-n" , 
                                        this.config.envName, 
                                        "--no-capture-output", 
                                        "alphadia", 
                                        "extract", 
                                        "--config", 
                                        path.join(workflow.output_directory.path, "config.yaml")
                                    ] , { env:{...process.env, PATH}, shell: true});
            run.pid = run.process.pid

            const stdoutTransform = lineBreakTransform();
            run.process.stdout.pipe(stdoutTransform).on('data', (data) => {
                run.std.push(data.toString())
            });

            const stderrTransform = lineBreakTransform();
            run.process.stderr.pipe(stderrTransform).on('data', (data) => {
                run.std.push(data.toString())
            });

            run.activePromise = new Promise((resolve, reject) => {
                run.process.on('close', (code) => {
                    run.process = null
                    run.code = code
                    resolve(code);
                });
            })

            console.log(run)

            return run
        }).catch((err) => {
            console.log(err)
            dialog.showMessageBox({
                type: 'error',
                title: 'Error while starting workflow',
                message: `Could not start workflow. ${err}`,
            }).catch((err) => {
                console.log(err)
            })
        })
    }
}
        
class DockerExecutionEngine extends BaseExecutionEngine {

    name = "DockerExecutionEngine"
    description = "Local or remote Docker containers are being used to run alphaDIA. Recommended for running alphaDIA in a containerized environment."
    valid_plattforms = [
        "win32",
        "linux",
        "darwin"
    ]

    checkAvailability(){
        return new Promise((resolve, reject) => {

            if (this.valid_plattforms.indexOf(process.platform) == -1){
                this.available = false
                this.error = "Plattform " + process.platform + " is not supported when using " + this.constructor.name
                resolve(this)
            }

            this.available = false
            this.error = "Docker not yet implemented"

            console.log(this.constructor.name + " status")
            console.log(this)
            resolve(this)
        })
    }
}

function hasWSL(){
    return new Promise((resolve, reject) => {
        exec("wsl --list", () => {}).on("exit", (code) => {
            if (code == 0){
                resolve()
            } else {
                reject("WSL not found")
            }
        })
    })
}

function hasWSLConda(){
    return new Promise((resolve, reject) => {
        exec("wsl bash -ic \"conda info --json\"", (err, stdout, stderr) => {
            if (err) {console.log(err); reject("Conda not found within WSL"); return;}
            const info = JSON.parse(stdout);
            resolve(info['conda_version'])
        })
    })
}

function hasWSLCondaEnv(envName){
    return new Promise((resolve, reject) => {
        exec("wsl bash -ic \"conda activate " + envName + "\"", (err, stdout, stderr) => {
            if (err) {console.log(err); reject("Conda environment " + envName + " not found within WSL"); return;}
            resolve()
        })
    })
}

function hasWSLAlphaPython(envName){
    return new Promise((resolve, reject) => {
        exec("wsl bash -ic \"conda activate " + envName + " && python -c \\\"print(__import__('sys').version)\\\"\"", (err, stdout, stderr) => {
            if (err) {console.log(err); reject("Python not found in environment " + envName + " within WSL"); return;}
            resolve(stdout.trim().split(" ")[0])
        })
    })
}

function hasWSLAlphaDIA(envName){
    return new Promise((resolve, reject) => {
        exec("wsl bash -ic \"conda activate " + envName + " && alphadia --version\"", (err, stdout, stderr) => {
            if (err) {console.log(err); reject("alphaDIA not found within WSL"); return;}
            resolve(stdout.trim())
        })
    })
}

class WSLExecutionEngine extends BaseExecutionEngine {

    name = "WSLExecutionEngine"
    description = "Windows subsystem for Linux (WSL) is being used to run alphaDIA. Recommended execution engine for Windows users."
    valid_plattforms = [
        "win32"
    ]

    checkAvailability(){
        return new Promise((resolve, reject) => {

            if (this.valid_plattforms.indexOf(process.platform) == -1){
                this.available = false
                this.error = "Plattform " + process.platform + " is not supported when using " + this.constructor.name
                resolve(this)
            }

            // check the status code of the command "wsl --list"

            return hasWSL().then(
                hasWSLConda
            ).then((conda_version) => {
                this.versions.push({"name": "conda", "version": conda_version})
            }).then(() => {
                return hasWSLCondaEnv(this.config.envName)
            }).then(() => {
                return hasWSLAlphaPython(this.config.envName)
            }).then((python_version) => {
                this.versions.push({"name": "python", "version": python_version})
            }).then(() => {
                return hasWSLAlphaDIA(this.config.envName)
            }).then((alphadia_version) => {
                this.versions.push({"name": "alphadia", "version": alphadia_version})
            }).then(() => {
                this.available = true
            }).catch((err) => {
                this.available = false
                this.error = err
            }).finally(() => {
                console.log(this.constructor.name + " status")
                console.log(this)
                resolve(this)
            })
        })
    }

    saveWorkflow(workflow){
        return new Promise((resolve, reject) => {
            const workflowFolder = workflow.output_directory.path
            const config = workflowToConfig(workflow)
            if (!fs.existsSync(workflowFolder)) {
                reject("Output folder " + workflowFolder + " does not exist.")
            }
            const configPath = path.join(workflowFolder, "config.yaml")
            // save config.yaml in workflow folder
            writeYamlFile.sync(configPath, config, {lineWidth: -1})
            resolve()
        })
    }

    startWorkflow(workflow){
        return this.saveWorkflow(workflow).then(() => {

            let run = {
                engine: this.constructor.name,
                name: workflow.name,
                path: workflow.output_directory.path,
                std: [],
                pid: null,
                code: -1,
                process : null,
                activePromise: null,
            }

            const winOutputPath = path.join(workflow.output_directory.path, "config.yaml")
            const wslOutputPath = "/mnt/" + winOutputPath.replace(/\\/g, "/").replace(":", "").toLowerCase()

            run.process = spawn("wsl", ["bash",
                                        "-ic",
                                        "\"conda run -n " + this.config.envName + " --no-capture-output alphadia extract -w --config " + wslOutputPath + "\""], { shell: true });
            run.pid = run.process.pid

            const stdoutTransform = lineBreakTransform();
            run.process.stdout.pipe(stdoutTransform).on('data', (data) => {
                run.std.push(data.toString())
            });

            const stderrTransform = lineBreakTransform();
            run.process.stderr.pipe(stderrTransform).on('data', (data) => {
                run.std.push(data.toString())
            });

            run.activePromise = new Promise((resolve, reject) => {
                run.process.on('close', (code) => {
                    run.process = null
                    run.code = code
                    resolve(code);
                });
            })

            console.log(run)

            return run
        }).catch((err) => {
            console.log(err)
            dialog.showMessageBox({
                type: 'error',
                title: 'Error while starting workflow',
                message: `Could not start workflow. ${err}`,
            }).catch((err) => {
                console.log(err)
            })
        })
    }



}

class ExecutionManager {

    initResolved = false;
    initPromise = null;

    runList = []

    constructor(profile){

        console.log(profile)

        this.executionEngines = [
            new CMDExecutionEngine(profile.config.CMDExecutionEngine || {}),
            new DockerExecutionEngine(profile.config.DockerExecutionEngine || {}),
            new WSLExecutionEngine(profile.config.WSLExecutionEngine || {}),
        ]

        // invoke checkAvailability() for all execution engines
        this.initPromise = Promise.all(this.executionEngines.map((engine) => {
            return engine.checkAvailability()
        })).then(() => {
            this.initResolved = true
        }).catch((err) => {
            console.log("Error while initializing ExecutionManager: " + err)
        })

    }

    getEngineStatus(){
        if (this.initResolved){
            return new Promise((resolve, reject) => {
                resolve(
                    this.executionEngines.map((engine) => {
                        return engine.getStatus()
                    })
                )
            })
        }
        return this.initPromise.then(() => {
            return this.getEngineStatus()
        })
    }

    startWorkflow(workflow, engineIdx){
        if (this.initResolved){

            const engine = this.executionEngines[engineIdx]
            return engine.startWorkflow(workflow).then((run) => {
                    console.log(run)
                    this.runList.push(run)
                    return run.activePromise
                }).catch((err) => {
                    console.log(err)
                })
        }
        return this.initPromise.then(() => {
            return this.startWorkflow(workflow, engineIdx)
        })
    }

    abortWorkflow(runIdx){
        console.log("Aborting workflow " + runIdx)
        console.log(this.runList)
        if (runIdx == -1){
            runIdx = this.runList.length - 1
        }
        console.log("runIdx: " + runIdx)
        if (this.runList[runIdx].pid != null){
            console.log(`Killing process ${this.runList[runIdx].pid}`)
            kill(this.runList[runIdx].pid);
        }
    }

    getOutputLength(runIdx){
        if (runIdx == -1){
            runIdx = this.runList.length - 1
        }
        return new Promise((resolve, reject) => {
            if (runIdx >= this.runList.length){
                reject("Run index out of bounds")
            } else if (runIdx == -1){
                resolve(0)
            }

            resolve(this.runList[runIdx].std.length)
        })
    }

    getOutputRows(runIdx, limit, offset){
        if (runIdx == -1){
            runIdx = this.runList.length - 1
        }
        return new Promise((resolve, reject) => {
            if (runIdx >= this.runList.length){
                reject("Run index out of bounds")
            } else if (runIdx == -1){
                resolve([])
            }
            const startIndex = offset
            const stopIndex = Math.min(offset + limit, this.runList[runIdx].std.length)
            resolve(this.runList[runIdx].std.slice(startIndex, stopIndex))
        })
    }
}

module.exports = {
    ExecutionManager
}
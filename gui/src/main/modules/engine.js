const { exec, spawn } = require('child_process');
const os = require('os');

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
            execPATH(`conda run -n ${envName} python --version` , condaPath, (err, stdout, stderr) => {
                if (err) {console.log(err); reject("Conda environment " + envName + " not found within WSL"); return;}
                resolve(stdout.trim())
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
                console.log(this.constructor.name + ": status = " + JSON.stringify(this))
                resolve(this)
            }).catch((err) => {
                this.available = false
                this.error = err
                console.log(this.constructor.name + ": status = " + JSON.stringify(this))
                resolve(this)
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

            console.log(this.constructor.name + ": status = " + JSON.stringify(this))
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
                this.versions.push({"name": "python", "version": "3.8.5"})
            }).then(() => {
                return hasWSLAlphaDIA(this.config.envName)
            }).then((alphadia_version) => {
                this.versions.push({"name": "alphadia", "version": alphadia_version})
            }).then(() => {
                this.available = true
                console.log(this.config)
                console.log(this.constructor.name + ": status = " + JSON.stringify(this))
                resolve(this)
            }).catch((err) => {
                this.available = false
                this.error = err
                console.log(this.constructor.name + ": status = " + JSON.stringify(this))
                resolve(this)
            })
        })
    }
}

class ExecutionManager {

    initResolved = false;
    initPromise = null;

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

}

module.exports = {
    ExecutionManager
}
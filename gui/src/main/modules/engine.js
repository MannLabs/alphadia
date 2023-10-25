const { exec, spawn } = require('child_process');

class BaseExecutionEngine {

    valid_plattforms = [
    ]

    status = {
        available: false,
        error: "",
        versions: {}
    }

    config = {}

    constructor(config){
        console.log(config)
        this.config = {...this.config, ...config}
    }

    checkAvailability(){
        // raise error if not implemented by subclass
        throw new Error("checkAvailability() not implemented by subclass")
    }
    
    getStatus(){
        return this.status
    }
}

class CMDExecutionEngine extends BaseExecutionEngine {

    valid_plattforms = [
        "win32",
        "linux",
        "darwin"
    ]
    checkAvailability(){
        return new Promise((resolve, reject) => {
            
            if (this.valid_plattforms.indexOf(process.platform) == -1){
                this.status.available
                this.status.error = "Plattform " + process.platform + " is not supported when using " + this.constructor.name
                resolve(this.status)
            }

            this.status.available = false
            this.status.error = "CMD not yet implemented"

            console.log(this.constructor.name + ": status = " + JSON.stringify(this.status))
            resolve(this.status)
        })
    }

}

class DockerExecutionEngine extends BaseExecutionEngine {

    valid_plattforms = [
        "win32",
        "linux",
        "darwin"
    ]

    checkAvailability(){
        return new Promise((resolve, reject) => {

            if (this.valid_plattforms.indexOf(process.platform) == -1){
                this.status.available = false
                this.status.error = "Plattform " + process.platform + " is not supported when using " + this.constructor.name
                resolve(this.status)
            }

            this.status.available = false
            this.status.error = "Docker not yet implemented"

            console.log(this.constructor.name + ": status = " + JSON.stringify(this.status))
            resolve(this.status)
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

    valid_plattforms = [
        "win32"
    ]

    checkAvailability(){
        return new Promise((resolve, reject) => {

            if (this.valid_plattforms.indexOf(process.platform) == -1){
                this.status.available = false
                this.status.error = "Plattform " + process.platform + " is not supported when using " + this.constructor.name
                resolve(this.status)
            }

            // check the status code of the command "wsl --list"

            return hasWSL().then(
                hasWSLConda
            ).then((conda_version) => {
                this.status.available = true
                this.status.versions['conda'] = conda_version
            }).then(() => {
                return hasWSLCondaEnv(this.config.envName)
            }).then(() => {
                this.status.available = true
                this.status.versions['environment'] = this.config.envName
            }).then(() => {
                return hasWSLAlphaDIA(this.config.envName)
            }).then((alphadia_version) => {
                this.status.available = true
                this.status.versions['alphadia'] = alphadia_version
            }).then(() => {
                console.log(this.config)
                console.log(this.constructor.name + ": status = " + JSON.stringify(this.status))
                resolve(this.status)
            }).catch((err) => {
                this.status.available = false
                this.status.error = err
                console.log(this.constructor.name + ": status = " + JSON.stringify(this.status))
                resolve(this.status)
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
                
                let status_array = {}
                this.executionEngines.forEach((engine) => {
                    status_array[engine.constructor.name] = engine.getStatus()
                })

                resolve(status_array)
            })
        }
        return this.initPromise.then(() => {
            return this.getEngineStatuss()
        })
    }
}

module.exports = {
    ExecutionManager
}
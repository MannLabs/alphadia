const { exec, spawn } = require('child_process');
var path = require('path');
const Transform = require('stream').Transform;
const StringDecoder = require('string_decoder').StringDecoder;
const { dialog } = require('electron')
const os = require('os');
const { proc } = require('node-os-utils');
var kill = require('tree-kill');
const { getCondaPath } = require('./condaUtils');

function testCommand(command, pathUpdate){
    const PATH = process.env.PATH + ":" + pathUpdate
    return new Promise((resolve, reject) => {
        exec(command, {env: {...process.env, PATH}}, () => {}).on('exit', (code) => {resolve(code)});
    });
}

const CondaEnvironment = class {

    pathUpdate = ""
    exists = {
        conda: false,
        python: false,
        alphadia: false
    }
    versions = {
        conda: "",
        python: "",
        alphadia: ""
    }
    ready = false;

    initPending = true;
    initPromise = null;

    std = [];
    pid = null;

    constructor(profile){
        this.profile = profile;

        this.initPromise = this.discoverCondaPATH().then((pathUpdate) => {
            this.pathUpdate = pathUpdate;
            this.exists.conda = true;

        }).then(() => {
            return this.checkCondaVersion().then((info) => {

                // executing conda run -n envName in an alreadt activated conda environment causes an error
                // this is a workaround
                // check if info exist and active_prefix is not null
                if (info != null){
                    if (info["active_prefix"] != null){
                        if (path.basename(info["active_prefix"]) == profile.config.conda.envName){
                            //dialog.showErrorBox("Conda environment already activated", "The conda environment " + this.envName + " is already activated. Please deactivate the environment and restart alphaDIA.")
                            return Promise.reject("Conda environment already activated")
                        }
                    }
                }
                return Promise.all([
                    this.checkPythonVersion(),
                    this.checkAlphadiaVersion(),
                ])
            }).catch((error) => {
                console.log("Conda not found", "Conda could not be found on your system. Please make sure conda is installed and added to your PATH." + error)
                dialog.showErrorBox("Conda not found", "Conda could not be found on your system. Please make sure conda is installed and added to your PATH." + error)
            })
        }).then(() => {
            this.ready = [this.exists.conda, this.exists.python, this.exists.alphadia].every(Boolean);
            this.initPending = false;

        }).catch((error) => {
            dialog.showErrorBox("Conda not found", "Conda could not be found on your system. Please make sure conda is installed and added to your PATH." + error)
        })
    }

    discoverCondaPATH(){
        return new Promise((resolve, reject) => {

            // 1st choice: conda is already in PATH
            // 2nd choice: conda path from profile setting is used
            // 3rd choice: default conda paths are tested
            const paths = ["", this.profile.config.conda.path, ...getCondaPath(os.userInfo().username, os.platform())]
            Promise.all(paths.map((path) => {
                return testCommand("conda", path)
                })).then((codes) => {
                    const index = codes.findIndex((code) => code == 0)

                    if (index == -1){
                        reject("conda not found")
                    } else {
                        resolve(paths[index])
                    }
            })
        })
    }

    checkCondaVersion(){
        return new Promise((resolve, reject) => {
            this.exec('conda info --json', (err, stdout, stderr) => {
                if (err) {console.log(err); reject(err); return;}
                const info = JSON.parse(stdout);
                this.versions.conda = info["conda_version"];
                this.exists.conda = true;
                resolve(info);
            });
        })
    }

    checkPythonVersion(){
        return new Promise((resolve, reject) => {

            this.exec(`conda run -n ${this.profile.config.conda.envName} python --version`, (err, stdout, stderr) => {
                if (err) {console.log(err); reject(err); return;}
                const versionPattern = /\d+\.\d+\.\d+/;
                const versionList = stdout.match(versionPattern);

                if (versionList == null){return;}
                if (versionList.length == 0){return;}

                this.versions.python = versionList[0];
                this.exists.python = true;
                resolve();
            });
        })
    }
    checkAlphadiaVersion(){
        return new Promise((resolve, reject) => {
            this.exec(`conda list -n ${this.profile.config.conda.envName} --json`, (err, stdout, stderr) => {
                if (err) {console.log(err); reject(err); return;}
                const info = JSON.parse(stdout);
                const packageInfo = info.filter((p) => p.name == "alphadia");
                if (packageInfo.length == 0){return;}

                this.versions.alphadia = packageInfo[0].version;
                this.exists.alphadia = true;
                resolve();
            });
        })
    }

    exec(command, callback){
        const PATH = process.env.PATH + ":" + this.pathUpdate
        exec(command, {env: {...process.env, PATH}}, callback);
    }

    buildEnvironmentStatus(){
        return {
            envName: this.profile.config.conda.envName,
            versions: this.versions,
            exists: this.exists,
            ready: this.ready
        }
    }

    getEnvironmentStatus(){
        if (this.initPending){
            return this.initPromise.then(() => {
                return this.buildEnvironmentStatus();
            })
        } else {
            return this.buildEnvironmentStatus();
        }
    }

    spawn(cmd){
        console.log(cmd)
        this.std = [];
        return new Promise((resolve, reject) => {
            if (!this.ready){
                reject("Environment not ready");
                return;
            }


            const PATH = process.env.PATH + ":" + this.pathUpdate
            const tokens = cmd.split(" ")
            const cmdp = spawn(tokens[0], tokens.slice(1), { env:{...process.env, PATH}, shell: true});

            const stdoutTransform = lineBreakTransform();
            cmdp.stdout.pipe(stdoutTransform).on('data', (data) => {
                this.std.push(data.toString())
            });

            const stderrTransform = lineBreakTransform();
            cmdp.stderr.pipe(stderrTransform).on('data', (data) => {
                this.std.push(data.toString())
            });

            cmdp.on('close', (code) => {
                resolve(code);
                return;
            });

            this.pid = cmdp.pid;

        })
    }

    kill(){
        if (this.pid != null){
            console.log(`Killing process ${this.pid}`)
            kill(this.pid);
        }

    }

    getOutputLength(){
        return this.std.length
    }

    getOutputRows(limit, offset){
        const startIndex = offset
        const stopIndex = Math.min(offset + limit, this.std.length)
        return this.std.slice(startIndex, stopIndex)
    }

}

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






module.exports = {
    CondaEnvironment
}

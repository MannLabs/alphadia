const fs = require("fs")
const path = require("path")
const { app, shell} = require("electron")
const { dialog } = require('electron')

const VERSION = "1.7.0"

const Profile = class {

    config = {
        "version": VERSION,
        "conda": {
            "envName": "alphadia",
            "path": ""
        },
        "clippy": false,
        "WSLExecutionEngine": {
            "envName": "alphadia",

        },
        "CMDExecutionEngine": {
            "envName": "alphadia",
            "condaPath": ""
        },
        "BundledExecutionEngine": {
            "binaryPath": ""
        },
    }

    constructor() {

        // check if profile exists
        if (!fs.existsSync(this.getProfilePath())) {
            this.saveProfile()
        } else {
            this.loadProfile()
        }
    }

    saveProfile() {
        fs.writeFileSync(this.getProfilePath(), JSON.stringify(this.config, null, 4))
    }

    loadProfile() {
        try {
            config = JSON.parse(fs.readFileSync(this.getProfilePath()))
        } catch (err) {
            dialog.showMessageBox({
                type: 'error',
                title: 'Error while loading profile',
                message: `Could not load profile. ${err}`,
            }).catch((err) => {
                console.log(err)
            })
            return
        }

        if (config.version !== VERSION) {
            dialog.showMessageBox({
                type: 'info',
                title: 'Found old alphaDIA profile',
                message: `Found old alphaDIA profile. Updating to version ${VERSION}`,
            }).catch((err) => {
                console.log(err)
            })
            this.saveProfile()
        } else {
            this.config = config
        }
    }

    getProfilePath() {
        return path.join(app.getPath("userData"), "profile.json")
    }

    openProfile() {
        shell.openPath(this.getProfilePath())
    }
}

module.exports = {
    Profile
}

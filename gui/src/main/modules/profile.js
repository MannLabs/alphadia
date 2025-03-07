const fs = require("fs")
const path = require("path")
const { app, shell, BrowserWindow} = require("electron")
const { dialog } = require('electron')

const VERSION = "1.10.1-dev0"

const Profile = class {

    config = {
        "version": VERSION,
        "conda": {
            "envName": "alphadia",
            "path": ""
        },
        "clippy": false,
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
        let loaded_config = {}

        try {
            loaded_config = JSON.parse(fs.readFileSync(this.getProfilePath()))
            console.log(loaded_config)
        } catch (err) {
            dialog.showMessageBox(
                BrowserWindow.getFocusedWindow(),
                {
                type: 'error',
                title: 'Error while loading profile',
                message: `Could not load profile. ${err}`,
            }).catch((err) => {
                console.log(err)
            })
            return
        }

        if (loaded_config.version !== VERSION) {
            dialog.showMessageBox(
                BrowserWindow.getFocusedWindow(),
                {
                type: 'info',
                title: 'Found old alphaDIA profile',
                message: `Found old alphaDIA profile. Updating to version ${VERSION}`,
            }).catch((err) => {
                console.log(err)
            })
            this.saveProfile()
        } else {
            this.config = loaded_config
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

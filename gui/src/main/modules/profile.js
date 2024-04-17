
const fs = require("fs")
const path = require("path")
const { app, shell} = require("electron")
const { dialog } = require('electron')

const Profile = class {

    config = {
        "version": "1.5.5",
        "conda": {
            "envName": "alpha",
            "path": ""
        },
        "clippy": false,
        "WSLExecutionEngine": {
            "envName": "alpha",
            
        },
        "CMDExecutionEngine": {
            "envName": "alpha",
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

        console.log(this.getProfilePath())
    }

    saveProfile() {
        fs.writeFileSync(this.getProfilePath(), JSON.stringify(this.config, null, 4))
    }

    loadProfile() {
        try {
            this.config = JSON.parse(fs.readFileSync(this.getProfilePath()))
        } catch (err) {
            dialog.showMessageBox({
                type: 'error',
                title: 'Error while loading profile',
                message: `Could not load profile. ${err}`,
            }).catch((err) => {
                console.log(err)
            })
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
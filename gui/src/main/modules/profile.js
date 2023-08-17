
const fs = require("fs")
const path = require("path")
const { app, shell} = require("electron")

const Profile = class {

    config = {
        "version": "1.3.0",
        "conda": {
            "envName": "alpha",
            "path": ""
        },
        "clippy": false,
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
        this.config = JSON.parse(fs.readFileSync(this.getProfilePath()))
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
function getCondaPath(username, platform){
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

module.exports = {
    getCondaPath
}

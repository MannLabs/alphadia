const { dialog } = require('electron')

/* returns a single folder */
function handleGetSingleFolder(window) {
    return (event) => {
        return dialog.showOpenDialog(window, {
            properties: ['openDirectory','createDirectory']
        }).then((result) => {
            if (result.canceled) {
                return ""
            } else {
                return result.filePaths[0]
            }
        }).catch((err) => {
            alert(err)
            return ""
        })
    }
}

/* returns an array of folders */
function handleGetMultipleFolders(window) {
    return (event) => {
        return dialog.showOpenDialog(window, {
            properties: ['openDirectory','multiSelections','createDirectory']
        }).then((result) => {
            if (result.canceled) {
                return []
            } else {
                return result.filePaths
            }
        }).catch((err) => {
            alert(err)
            return []
        })
    }
}

/* returns a single file */
function handleGetSingleFile(window) {
    return (event) => {
        return dialog.showOpenDialog(window, {
            properties: ['openFile','createDirectory']
        }).then((result) => {
            if (result.canceled) {
                return ""
            } else {
                return result.filePaths[0]
            }
        }).catch((err) => {
            alert(err)
            return ""
        })
    }
}

/* returns an array of files */
function handleGetMultipleFiles(window) {
    return (event) => {
        return dialog.showOpenDialog(window, {
            properties: ['openFile','multiSelections','createDirectory']
        }).then((result) => {
            if (result.canceled) {
                return []
            } else {
                return result.filePaths
            }
        }).catch((err) => {
            alert(err)
            return []
        })
    }
}

module.exports = {
    handleGetSingleFolder,
    handleGetMultipleFolders,
    handleGetSingleFile,
    handleGetMultipleFiles
}
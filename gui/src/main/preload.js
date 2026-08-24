const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
    getSingleFolder: () => ipcRenderer.invoke('get-single-folder'),
    getMultipleFolders: () => ipcRenderer.invoke('get-multiple-folders'),
    getSingleFile: () => ipcRenderer.invoke('get-single-file'),
    getMultipleFiles: () => ipcRenderer.invoke('get-multiple-files'),
    getUtilisation: () => ipcRenderer.invoke('get-utilisation'),
    getWorkflows: () => ipcRenderer.invoke('get-workflows'),

    openLink: (url) => ipcRenderer.send('open-link', url),

    onThemeChange: (callback) => ipcRenderer.on('theme-change', callback),

    getEngineStatus: () => ipcRenderer.invoke('get-engine-status'),
    startWorkflowNew: (workflow, engineIdx) => ipcRenderer.invoke('start-workflow-new', workflow, engineIdx),
    abortWorkflowNew: (runIdx) => ipcRenderer.invoke('abort-workflow-new', runIdx),
    getOutputRowsNew: (runIdx, {limit, offset}) => ipcRenderer.invoke('get-output-rows-new', runIdx, {limit, offset}),
    getOutputLengthNew: (runIdx) => ipcRenderer.invoke('get-output-length-new', runIdx),

})

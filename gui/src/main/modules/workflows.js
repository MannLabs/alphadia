const { app, dialog } = require('electron')
const fs = require('fs');
const path = require('path');

function discoverWorkflows(window){

    directory = app.getAppPath() + "/workflows";
    console.log(directory)

    const files = fs.readdirSync(directory)
    return files.reduce((acc, file) => {
        try {
            const workflow = JSON.parse(fs.readFileSync(path.join(directory, file)))
            workflow.name = path.basename(file, '.json')
            validateWorkflow(workflow)
            return [...acc, workflow]
        } catch (err) {
            dialog.showMessageBox(window, {
                type: 'warning',
                title: 'Workflow Failed to Load',
                message: `Could not load workflow ${file}. ${err.message}`,
            }).catch((err) => {
                console.log(err)
            })
            return acc
        }
    }, [])
}

function validateWorkflow(workflow) {

    // make sure workflow has the following fields: [library, fasta, files, config]
    if (!workflow.hasOwnProperty('library')) {
        throw new Error('Workflow does not have a library field.')
    }
    if (!workflow.hasOwnProperty('fasta')) {
        throw new Error('Workflow does not have a fasta field.')
    }
    if (!workflow.hasOwnProperty('files')) {
        throw new Error('Workflow does not have a files field.')
    }
    if (!workflow.hasOwnProperty('config')) {
        throw new Error('Workflow does not have a config field.')
    }

    // make sure library has the following fields: [active, path]
    if (!workflow.library.hasOwnProperty('active')) {
        throw new Error('Workflow library does not have an active field.')
    }
    if (!workflow.library.hasOwnProperty('path')) {
        throw new Error('Workflow library does not have a path field.')
    }

    // make sure fasta has the following fields: [active, path]
    if (!workflow.fasta.hasOwnProperty('active')) {
        throw new Error('Workflow fasta does not have an active field.')
    }
    if (!workflow.fasta.hasOwnProperty('path')) {
        throw new Error('Workflow fasta does not have a path field.')
    }

    // make sure files has the following fields: [active, path]
    if (!workflow.files.hasOwnProperty('active')) {
        throw new Error('Workflow files does not have an active field.')
    }
    if (!workflow.files.hasOwnProperty('path')) {
        throw new Error('Workflow files does not have a path field.')
    }

    // make sure config is an array
    if (!Array.isArray(workflow.config)) {
        throw new Error('Workflow config is not an array.')
    }

}

function workflowToConfig(workflow) {

    let output = {name: workflow.name}

    if (workflow.library.path != "") {
        output["library"] = workflow.library.path
    }
    
    if (workflow.fasta.path != "") {
        output["fasta"] = workflow.fasta.path
    }

    if (workflow.files.path != "") {
        output["files"] = workflow.files.path
    }

    if (workflow.output.path != "") {
        output["output"] = workflow.output.path
    }
    
    workflow.config.forEach((config) => {
        output[config.id] = {}
        config.parameters.forEach((parameter) => {
            output[config.id][parameter.id] = parameter.value
        })
    })

    return output
}    

module.exports = {
    discoverWorkflows,
    validateWorkflow,
    workflowToConfig
}

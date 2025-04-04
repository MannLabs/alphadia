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
            initializeWorkflow(workflow)
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
    if (!workflow.hasOwnProperty('fasta_list')) {
        throw new Error('Workflow does not have a fasta field.')
    }
    if (!workflow.hasOwnProperty('raw_path_list')) {
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
    if (!workflow.fasta_list.hasOwnProperty('active')) {
        throw new Error('Workflow fasta does not have an active field.')
    }
    if (!workflow.fasta_list.hasOwnProperty('path')) {
        throw new Error('Workflow fasta does not have a path field.')
    }

    // make sure files has the following fields: [active, path]
    if (!workflow.raw_path_list.hasOwnProperty('active')) {
        throw new Error('Workflow files does not have an active field.')
    }
    if (!workflow.raw_path_list.hasOwnProperty('path')) {
        throw new Error('Workflow files does not have a path field.')
    }

    // make sure config is an array
    if (!Array.isArray(workflow.config)) {
        throw new Error('Workflow config is not an array.')
    }

}

function initializeWorkflow(workflow) {
    // Process each config section
    workflow.config.forEach((config) => {
        // Process regular parameters
        if (config.parameters) {
            config.parameters.forEach((parameter) => {
                // Only set value to default if value is not defined
                if (parameter.value === undefined) {
                    parameter.value = parameter.default;
                }
            });
        }

        // Process advanced parameters
        if (config.parameters_advanced) {
            config.parameters_advanced.forEach((parameter) => {
                // Only set value to default if value is not defined
                if (parameter.value === undefined) {
                    parameter.value = parameter.default;
                }
            });
        }
    });

    return workflow;
}

function workflowToConfig(workflow) {

    let output = {workflow_name: workflow.name}

    if (workflow.library.path != "") {
        output["library_path"] = workflow.library.path
    }

    if (workflow.fasta_list.path != "") {
        output["fasta_paths"] = workflow.fasta_list.path
    }

    if (workflow.raw_path_list.path != "") {
        output["raw_paths"] = workflow.raw_path_list.path
    }

    if (workflow.output_directory.path != "") {
        output["output_directory"] = workflow.output_directory.path
    }

    workflow.config.forEach((config) => {
        output[config.id] = {}
        config.parameters.forEach((parameter) => {
            output[config.id][parameter.id] = parameter.value
        })
        config.parameters_advanced.forEach((parameter) => {
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

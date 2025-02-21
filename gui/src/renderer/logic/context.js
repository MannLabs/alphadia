import React, { useReducer } from "react";

const initialMethod = {

    library: {
        active: false,
        path: ""
    },
    fasta_list: {
        active: false,
        path: [
        ]
    },
    raw_path_list: {
        active: false,
        path: [
        ]
    },
    output_directory: {
        active: true,
        path: ""
    },
    config: [
    ]
}

export function methodReducer(method, action) {

    switch (action.type) {
        case 'updateLibrary':
            return {...method, library: {...method.library, path: action.path}}

        case 'updateFasta':
            return {...method, fasta_list: {...method.fasta_list, path: action.path}}

        case 'appendFasta':
            return {...method, fasta_list: {...method.fasta_list, path: method.fasta_list.path.concat(action.path)}}

        case 'updateFiles':
            return {...method, raw_path_list: {...method.raw_path_list, path: action.path}}

        case 'appendFiles':
            return {...method, raw_path_list: {...method.raw_path_list, path: method.raw_path_list.path.concat(action.path)}}

        case 'updateParameter':
            const new_config = method.config.map((parameterGroup) => {
                if (parameterGroup.id === action.parameterGroupId) {
                    const new_parameters = parameterGroup.parameters.map((parameter) => {
                        if (parameter.id === action.parameterId) {
                            return {...parameter, value: action.value}
                        } else {
                            return parameter
                        }
                    })
                    return {...parameterGroup, parameters: new_parameters}
                } else {
                    return parameterGroup
                }
            })

            return {...method, config: new_config}

        case 'updateParameterAdvanced':
            const new_config_advanced = method.config.map((parameterGroup) => {
                if (parameterGroup.id === action.parameterGroupId) {
                    const new_parameters_advanced = parameterGroup.parameters_advanced.map((parameter) => {
                        if (parameter.id === action.parameterId) {
                            return {...parameter, value: action.value}
                        } else {
                            return parameter
                        }
                    })
                    return {...parameterGroup, parameters_advanced: new_parameters_advanced}
                } else {
                    return parameterGroup
                }
            })
            return {...method, config: new_config_advanced}

        case "updateWorkflow":
            return {...method, ...action.workflow}

        case "updateOutput":
            return {...method, output_directory: {...method.output_directory, path: action.path}}
        default:
            throw new Error(`Unhandled action type: ${action.type}`);
    }
}

const MethodContext = React.createContext(null);
export function useMethod() {
    return React.useContext(MethodContext);
}

const MethodDispatchContext = React.createContext(null);
export function useMethodDispatch() {
    return React.useContext(MethodDispatchContext);
}

export function MethodProvider({ children }) {

    const [method, dispatch] = useReducer(
        methodReducer,
        initialMethod
    );

    return (
    <MethodContext.Provider value={method}>
      <MethodDispatchContext.Provider value={dispatch}>
        {children}
      </MethodDispatchContext.Provider>
    </MethodContext.Provider>
  );
}

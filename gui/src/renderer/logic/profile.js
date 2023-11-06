import React, { useReducer } from "react";

const initialProfile = {

    activeIdx: -1,
    executionEngines: [],

    activeWorkflowIdx: -1,
    workflows: [],

    running: false,
}

export function profileReducer(profile, action) {

    switch (action.type) {
        case 'setExecutionEngineIdx':
            if (action.idx >= profile.executionEngines.length) {
                throw new Error(`Invalid execution engine index: ${action.idx}`);
            }

            if (! profile.executionEngines[action.idx].available) {
                throw new Error(`Execution engine not available: ${profile.executionEngines[action.idx].name}`);
            }

            return {...profile, activeIdx: action.idx}

        case 'setExecutionEngines':
            return {...profile, executionEngines: action.executionEngines}

        case 'setWorkflowIdx':
            if (action.idx >= profile.workflows.length) {
                throw new Error(`Invalid workflow index: ${action.idx}`);
            }

            return {...profile, activeWorkflowIdx: action.idx}

        case 'setWorkflows':
            return {...profile, workflows: action.workflows}

        case 'setRunning':
            return {...profile, running: action.running}

        default:
            throw new Error(`Unhandled action type: ${action.type}`);
    }
}

const ProfileContext = React.createContext(null);
export function useProfile() {
    return React.useContext(ProfileContext);
}

const ProfileDispatchContext = React.createContext(null);
export function useProfileDispatch() {
    return React.useContext(ProfileDispatchContext);
}

export function ProfileProvider({ children }) {

    const [profile, dispatch] = useReducer(
        profileReducer,
        initialProfile
    );

    return (
    <ProfileContext.Provider value={profile}>
      <ProfileDispatchContext.Provider value={dispatch}>
        {children}
      </ProfileDispatchContext.Provider>
    </ProfileContext.Provider>
  );
}
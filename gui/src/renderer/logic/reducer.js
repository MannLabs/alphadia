
export function methodReducer(method, action) {

    switch (action.type) {
        case 'set':
            return action.payload;
        case 'update':
            return {...method, ...action.payload};
        case 'updateLibrary':
            return {...method, library: {...method.library, ...action.payload}};
        default:
            throw new Error(`Unhandled action type: ${action.type}`);
    }
}

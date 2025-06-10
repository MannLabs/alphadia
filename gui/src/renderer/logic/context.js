import React, { useReducer, createContext, useContext, useState, useEffect, useCallback } from "react";

// GitHub Data Provider
// Cache constants
const GITHUB_CACHE_KEY = 'alphax_github_cache';
const CACHE_EXPIRY = 60 * 60 * 1000; // 1 hour in milliseconds

// Repositories to track
const REPOSITORIES = [
    {url: "https://api.github.com/repos/MannLabs/alphadia/issues", name: "AlphaDIA"},
    {url: "https://api.github.com/repos/MannLabs/alphabase/issues", name: "AlphaBase"},
    {url: "https://api.github.com/repos/MannLabs/alphapept/issues", name: "AlphaPept"},
    {url: "https://api.github.com/repos/MannLabs/alphatims/issues", name: "AlphaTims"},
    {url: "https://api.github.com/repos/MannLabs/alphapeptdeep/issues", name: "AlphaPeptDeep"},
];

// GitHub Context
const GitHubContext = createContext(null);

// Helper functions for GitHub data fetching
function fetchGithubReleases(repos) {
    return Promise.all(repos.map(repo => {
        const releasesUrl = repo.url.replace('/issues', '/releases');
        return fetch(releasesUrl).then((res) => {
            if (!res.ok) {
                throw new Error(`HTTP error ${res.status} for ${releasesUrl}`);
            } else {
                return res.json();
            }
        }).then(releases => {
            // Filter out pre-releases
            const stableReleases = releases.filter(release => !release.prerelease);
            return {
                name: repo.name,
                releases: stableReleases
            }
        }).catch(err => {
            console.error(err);
            return { name: repo.name, releases: [] }
        })
    }));
}

function fetchGithubIssues(repos) {
    return Promise.all(repos.map(repo => {
        return fetch(repo.url).then((res) => {
            if (!res.ok) {
                throw new Error(`HTTP error ${res.status} for ${repo.url}`);
            } else {
                return res.json();
            }
        }).then(issues => {
            return {
                name: repo.name,
                issues: issues
            }
        }).catch(err => {
            console.error(err);
            return { name: repo.name, issues: [] }
        })
    }));
}

function combineItemsAndSort(issues, releases) {
    // Process releases
    const processedReleases = releases.flatMap(repo =>
        repo.releases.map(release => ({
            name: repo.name,
            title: `${repo.name} ${release.tag_name}`,
            url: release.html_url,
            updated_at: new Date(release.published_at || release.created_at),
            days_ago: Math.floor((new Date() - new Date(release.published_at || release.created_at)) / (1000 * 60 * 60 * 24)),
            hours_ago: Math.floor((new Date() - new Date(release.published_at || release.created_at)) / (1000 * 60 * 60)) % 24,
            type: 'release',
            tagName: release.tag_name,
            body: release.body || ''
        }))
    );

    // Process issues
    const processedIssues = issues.flatMap(repo =>
        (repo.issues || []).map(issue => ({
            name: repo.name,
            title: issue.title,
            url: issue.html_url,
            updated_at: new Date(issue.updated_at),
            days_ago: Math.floor((new Date() - new Date(issue.updated_at)) / (1000 * 60 * 60 * 24)),
            hours_ago: Math.floor((new Date() - new Date(issue.updated_at)) / (1000 * 60 * 60)) % 24,
            type: 'issue',
            number: issue.number,
            state: issue.state
        }))
    );

    // Combine and sort
    return {
        combinedItems: [...processedIssues, ...processedReleases]
            .sort((a, b) => b.updated_at - a.updated_at)
            .slice(0, 20),
        releases: processedReleases.sort((a, b) => b.updated_at - a.updated_at),
        issues: processedIssues.sort((a, b) => b.updated_at - a.updated_at),
        byRepo: REPOSITORIES.reduce((acc, repo) => {
            acc[repo.name] = {
                releases: processedReleases.filter(item => item.name === repo.name),
                issues: processedIssues.filter(item => item.name === repo.name)
            };
            return acc;
        }, {})
    };
}

// Cache functions
function getCache() {
    try {
        const cachedData = localStorage.getItem(GITHUB_CACHE_KEY);
        if (cachedData) {
            const { data, timestamp } = JSON.parse(cachedData);
            // Check if cache is still valid
            if (Date.now() - timestamp < CACHE_EXPIRY) {
                return { data, timestamp };
            }
        }
    } catch (error) {
        console.error('Error reading from cache:', error);
    }
    return null;
}

function setCache(data) {
    try {
        const timestamp = Date.now();
        const cacheObject = {
            data,
            timestamp
        };
        localStorage.setItem(GITHUB_CACHE_KEY, JSON.stringify(cacheObject));
        return timestamp;
    } catch (error) {
        console.error('Error writing to cache:', error);
        return Date.now();
    }
}

// Format the last updated time
function formatLastUpdated(timestamp) {
    if (!timestamp) return '';

    const now = Date.now();
    const diff = now - timestamp;

    // If less than a minute
    if (diff < 60 * 1000) {
        return 'just now';
    }

    // If less than an hour
    if (diff < 60 * 60 * 1000) {
        const minutes = Math.floor(diff / (60 * 1000));
        return `${minutes} minute${minutes !== 1 ? 's' : ''} ago`;
    }

    // If less than a day
    if (diff < 24 * 60 * 60 * 1000) {
        const hours = Math.floor(diff / (60 * 60 * 1000));
        return `${hours} hour${hours !== 1 ? 's' : ''} ago`;
    }

    // Otherwise show days
    const days = Math.floor(diff / (24 * 60 * 60 * 1000));
    return `${days} day${days !== 1 ? 's' : ''} ago`;
}

// Method context
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

// GitHub Provider
export function useGitHub() {
    const context = useContext(GitHubContext);
    if (!context) {
        throw new Error('useGitHub must be used within a GitHubProvider');
    }
    return context;
}

export function GitHubProvider({ children }) {
    const [githubData, setGithubData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState(null);

    const fetchData = useCallback(async () => {
        setLoading(true);

        try {
            // Fetch both issues and releases
            const [issues, releases] = await Promise.all([
                fetchGithubIssues(REPOSITORIES),
                fetchGithubReleases(REPOSITORIES)
            ]);

            // Process and combine the data
            const processedData = combineItemsAndSort(issues, releases);

            // Update state and cache
            setGithubData(processedData);
            const timestamp = setCache(processedData);
            setLastUpdated(timestamp);
        } catch (error) {
            console.error("Error fetching GitHub data:", error);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        // First check cache
        const cachedData = getCache();

        if (cachedData) {
            setGithubData(cachedData.data);
            setLastUpdated(cachedData.timestamp);
            setLoading(false);
        } else {
            // If no valid cache, fetch from API
            fetchData();
        }
    }, [fetchData]);

    const contextValue = {
        githubData,
        loading,
        lastUpdated,
        formatLastUpdated,
        refreshData: fetchData
    };

    return (
        <GitHubContext.Provider value={contextValue}>
            {children}
        </GitHubContext.Provider>
    );
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

// Combined provider
export function AppProviders({ children }) {
    return (
        <GitHubProvider>
            <MethodProvider>
                {children}
            </MethodProvider>
        </GitHubProvider>
    );
}

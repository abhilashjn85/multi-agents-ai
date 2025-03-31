/**
 * WorkflowEditor class for managing the anomaly detection workflow visualization and editing
 */
class WorkflowEditor {
    /**
     * Create a new WorkflowEditor instance
     * @param {string} containerId - ID of the HTML element to render the network in
     * @param {Object} options - Configuration options
     */
    constructor(containerId, options) {
        this.containerId = containerId;
        this.options = options || {};
        this.agents = this.options.agents || [];
        this.agentsMap = {};

        // Create a map of agents by ID for quick lookup
        this.agents.forEach(agent => {
            this.agentsMap[agent.id] = agent;
        });

        // Initialize network data
        this.nodes = new vis.DataSet([]);
        this.edges = new vis.DataSet([]);

        // Network configuration
        this.networkOptions = {
            nodes: {
                shape: 'box',
                shapeProperties: {
                    borderRadius: 6
                },
                font: {
                    size: 14,
                    face: 'Arial'
                },
                margin: 10,
                shadow: {
                    enabled: true,
                    size: 5,
                    x: 2,
                    y: 2
                }
            },
            edges: {
                arrows: {
                    to: { enabled: true, scaleFactor: 0.8 }
                },
                smooth: {
                    type: 'curvedCW',
                    roundness: 0.2
                },
                font: {
                    size: 12
                }
            },
            physics: {
                enabled: true,
                barnesHut: {
                    gravitationalConstant: -5000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04,
                    damping: 0.09
                },
                stabilization: {
                    enabled: true,
                    iterations: 1000,
                    updateInterval: 100
                }
            },
            interaction: {
                hover: true,
                dragNodes: true,
                dragView: true,
                zoomView: true
            },
            manipulation: {
                enabled: true,
                addNode: false,
                addEdge: (data, callback) => {
                    // Show connection configuration modal
                    $('#connection-from').val(data.from);
                    $('#connection-to').val(data.to);
                    $('#connection-label').val('');
                    $('#connection-type').val('default');
                    $('#connection-condition').val('');
                    $('#conditional-options').hide();

                    // Show the modal
                    $('#connectionModal').modal('show');

                    // Setup callback for when the connection is saved
                    $('#save-connection').off('click').on('click', () => {
                        const label = $('#connection-label').val();
                        const type = $('#connection-type').val();
                        const condition = $('#connection-condition').val();

                        data.label = label;
                        data.type = type;

                        if (type === 'conditional' && condition) {
                            data.condition = condition;
                        }

                        callback(data);
                        $('#connectionModal').modal('hide');
                    });

                    // Setup callback for when the modal is dismissed
                    $('#connectionModal').off('hidden.bs.modal').on('hidden.bs.modal', () => {
                        callback(null); // Cancel edge creation
                    });
                },
                editEdge: true,
                deleteEdge: true,
                deleteNode: true
            }
        };

        // Create the network
        this.container = document.getElementById(containerId);
        this.network = new vis.Network(this.container, { nodes: this.nodes, edges: this.edges }, this.networkOptions);

        // Setup event handlers
        this._setupEventHandlers();

        // Load workflow data if provided
        if (this.options.workflow) {
            this._loadWorkflow(this.options.workflow);
        } else {
            // Create a default layout if no workflow is provided
            this._createDefaultLayout();
        }
    }

    /**
     * Create a default layout for new workflows
     * @private
     */
    _createDefaultLayout() {
        // If no workflow is provided, create a simple default layout
        // positioning the Data Understanding Specialist as the first node
        const dataUnderstandingAgent = this.agents.find(agent =>
            agent.role.includes("Data Understanding") || agent.name.includes("Data Understanding"));

        if (dataUnderstandingAgent) {
            // Add the first node in the center
            const centerX = this.container.clientWidth / 2;
            const centerY = this.container.clientHeight / 3;
            this.addAgentNode(dataUnderstandingAgent.id, centerX, centerY);
        }
    }

    /**
     * Setup event handlers for the network
     * @private
     */
    _setupEventHandlers() {
        // Handle node selection
        this.network.on('selectNode', (params) => {
            if (params.nodes.length === 1) {
                const nodeId = params.nodes[0];
                const node = this.nodes.get(nodeId);
                const agent = this.agentsMap[node.agentId];

                // Populate node configuration panel
                $('#selected-node-id').val(nodeId);
                $('#node-agent-name').val(agent.name);
                $('#node-agent-goal').val(agent.goal);
                $('#node-max-iterations').val(node.maxIterations || 5);
                $('#node-communication-threshold').val(node.communicationThreshold || 0.7);
                $('#node-verbose').prop('checked', node.verbose !== false);
                $('#node-allow-delegation').prop('checked', node.allowDelegation !== false);

                // Populate task fields
                $('#node-task-description').val(node.task ? node.task.description : '');
                $('#node-task-output').val(node.task ? node.task.expectedOutput : '');

                // Show the configuration panel
                $('#node-config').show();
            } else {
                $('#node-config').hide();
            }
        });

        // Hide config panel when clicking on canvas
        this.network.on('deselectNode', () => {
            $('#node-config').hide();
        });

        // Handle double-click on node (show detailed view)
        this.network.on('doubleClick', (params) => {
            if (params.nodes.length === 1) {
                const nodeId = params.nodes[0];
                const node = this.nodes.get(nodeId);
                const agent = this.agentsMap[node.agentId];

                // Show detailed view of the agent (could open a modal or panel)
                alert(`Agent: ${agent.name}\nRole: ${agent.role}\nGoal: ${agent.goal}\nBackstory: ${agent.backstory}`);
            }
        });

        // Handle adding new connections
        this.network.on('click', (params) => {
            if (params.nodes.length === 0 && params.edges.length === 0) {
                // Clicked on empty space - deselect all
                this.network.unselectAll();
            }
        });
    }

    /**
     * Load a workflow into the editor
     * @param {Object} workflow - Workflow data object
     * @private
     */
    _loadWorkflow(workflow) {
        console.log("Loading workflow:", workflow);

        // Clear existing data
        this.nodes.clear();
        this.edges.clear();

        // Set workflow ID and name
        $('#workflow-id').data('workflow-id', workflow.id);
        $('#workflow-name').val(workflow.name);
        $('#workflow-process-type').val(workflow.process_type);
        $('#workflow-max-iterations').val(workflow.max_iterations);

        // Load agent nodes
        if (workflow.agent_ids && workflow.agent_ids.length > 0) {
            // Create a map of tasks by agent ID
            const tasksByAgentId = {};
            if (workflow.tasks && workflow.tasks.length > 0) {
                workflow.tasks.forEach(task => {
                    tasksByAgentId[task.agent_id] = task;
                });
            }

            // Position calculation
            const numAgents = workflow.agent_ids.length;
            const radius = Math.min(this.container.clientWidth, this.container.clientHeight) * 0.4;
            const centerX = this.container.clientWidth / 2;
            const centerY = this.container.clientHeight / 2;

            // Add nodes
            const nodeIdMap = {}; // Map agent IDs to node IDs

            workflow.agent_ids.forEach((agentId, index) => {
                const agent = this.agentsMap[agentId];
                if (agent) {
                    const angle = (2 * Math.PI * index) / numAgents;
                    const x = centerX + radius * Math.cos(angle);
                    const y = centerY + radius * Math.sin(angle);

                    // Get the task for this agent if it exists
                    const task = tasksByAgentId[agentId];

                    // Add the node
                    const nodeId = this.addAgentNode(agentId, x, y, task);
                    nodeIdMap[agentId] = nodeId;
                }
            });

            // Add connections between nodes
            if (workflow.connections && workflow.connections.length > 0) {
                workflow.connections.forEach(connection => {
                    if (nodeIdMap[connection.source] && nodeIdMap[connection.target]) {
                        this.edges.add({
                            from: nodeIdMap[connection.source],
                            to: nodeIdMap[connection.target],
                            label: connection.label || '',
                            type: connection.type || 'default',
                            condition: connection.conditions ? JSON.stringify(connection.conditions) : '',
                            arrows: {
                                to: { enabled: true, scaleFactor: 0.8 }
                            },
                            color: this._getEdgeColor(connection.type)
                        });
                    }
                });
            } else if (workflow.agent_ids.length > 1) {
                // If no connections are defined but we have multiple agents,
                // create default sequential connections
                for (let i = 0; i < workflow.agent_ids.length - 1; i++) {
                    const sourceId = nodeIdMap[workflow.agent_ids[i]];
                    const targetId = nodeIdMap[workflow.agent_ids[i + 1]];

                    if (sourceId && targetId) {
                        this.edges.add({
                            from: sourceId,
                            to: targetId,
                            arrows: {
                                to: { enabled: true, scaleFactor: 0.8 }
                            },
                            color: this._getEdgeColor('default')
                        });
                    }
                }
            }
        }

        // Fit the network to view all nodes
        this.fitToScreen();
    }

    /**
     * Get the color for an edge based on its type
     * @param {string} type - Edge type
     * @returns {Object} Color configuration
     * @private
     */
    _getEdgeColor(type) {
        switch (type) {
            case 'conditional':
                return { color: '#e67e22', highlight: '#d35400' };
            case 'async':
                return { color: '#3498db', highlight: '#2980b9' };
            default:
                return { color: '#2ecc71', highlight: '#27ae60' };
        }
    }

    /**
     * Add a new agent node to the workflow
     * @param {string} agentId - ID of the agent to add
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {Object} task - Optional task definition
     * @returns {string} ID of the created node
     */
    addAgentNode(agentId, x, y, task = null) {
        const agent = this.agentsMap[agentId];
        if (!agent) return null;

        // Generate a unique node ID
        const nodeId = 'node_' + agentId + '_' + new Date().getTime();

        // Create the node
        const node = {
            id: nodeId,
            label: agent.name,
            title: agent.role,
            agentId: agentId,
            x: x,
            y: y,
            color: {
                background: '#ecf0f1',
                border: '#3498db',
                highlight: {
                    background: '#d6eaf8',
                    border: '#2980b9'
                }
            },
            maxIterations: agent.max_iterations || 5,
            communicationThreshold: agent.communication_threshold || 0.7,
            verbose: agent.verbose !== false,
            allowDelegation: agent.allow_delegation !== false
        };

        // Add task if provided
        if (task) {
            node.task = {
                description: task.description,
                expectedOutput: task.expected_output,
                context: task.context || {}
            };
        }

        // Add the node to the network
        this.nodes.add(node);

        return nodeId;
    }

    /**
     * Update the configuration of a node
     * @param {string} nodeId - ID of the node to update
     * @param {Object} config - New configuration
     */
    updateNodeConfig(nodeId, config) {
        const node = this.nodes.get(nodeId);
        if (!node) return;

        // Update node configuration
        node.maxIterations = parseInt(config.maxIterations);
        node.communicationThreshold = parseFloat(config.communicationThreshold);
        node.verbose = config.verbose;
        node.allowDelegation = config.allowDelegation;

        // Update task
        if (!node.task) {
            node.task = {};
        }

        node.task.description = config.task.description;
        node.task.expectedOutput = config.task.expectedOutput;

        // Update the node in the network
        this.nodes.update(node);
    }

    /**
     * Update a connection between nodes
     * @param {Object} connectionData - Connection data
     */
    updateConnection(connectionData) {
        // Check if the connection already exists
        const existingEdges = this.edges.get({
            filter: edge => edge.from === connectionData.from && edge.to === connectionData.to
        });

        if (existingEdges.length > 0) {
            // Update existing edge
            const edge = existingEdges[0];
            edge.label = connectionData.label || '';
            edge.type = connectionData.type || 'default';
            edge.condition = connectionData.condition || '';
            edge.color = this._getEdgeColor(connectionData.type);

            this.edges.update(edge);
        } else {
            // Add new edge
            this.edges.add({
                from: connectionData.from,
                to: connectionData.to,
                label: connectionData.label || '',
                type: connectionData.type || 'default',
                condition: connectionData.condition || '',
                color: this._getEdgeColor(connectionData.type),
                arrows: {
                    to: { enabled: true, scaleFactor: 0.8 }
                }
            });
        }
    }

    /**
     * Save the current workflow
     * @param {string} name - Workflow name
     * @param {string} processType - Process type (sequential or parallel)
     * @param {number} maxIterations - Maximum iterations
     */
    saveWorkflow(name, processType, maxIterations) {
        // Get all nodes and edges
        const nodes = this.nodes.get();
        const edges = this.edges.get();

        // Create agent_ids array
        const agentIds = nodes.map(node => node.agentId);

        // Create tasks array
        const tasks = nodes.map(node => {
            const task = {
                id: 'task_' + node.id,
                agent_id: node.agentId,
                description: node.task ? node.task.description : '',
                expected_output: node.task ? node.task.expectedOutput : ''
            };

            // Add dependencies based on incoming edges
            const incomingEdges = edges.filter(edge => edge.to === node.id);
            if (incomingEdges.length > 0) {
                task.depends_on = incomingEdges.map(edge => {
                    const sourceNode = this.nodes.get(edge.from);
                    return sourceNode ? sourceNode.agentId : null;
                }).filter(id => id !== null);
            }

            return task;
        });

        // Create connections array
        const connections = edges.map(edge => {
            const sourceNode = this.nodes.get(edge.from);
            const targetNode = this.nodes.get(edge.to);

            if (!sourceNode || !targetNode) return null;

            return {
                source: sourceNode.agentId,
                target: targetNode.agentId,
                label: edge.label || '',
                type: edge.type || 'default',
                conditions: edge.condition ? { expression: edge.condition } : {}
            };
        }).filter(conn => conn !== null);

        // Create workflow object
        const workflow = {
            name: name,
            description: 'Anomaly detection workflow',
            process_type: processType,
            max_iterations: parseInt(maxIterations),
            agent_ids: agentIds,
            tasks: tasks,
            connections: connections
        };

        // Get existing workflow ID if available
        const workflowId = $('#workflow-id').data('workflow-id');
        if (workflowId) {
            workflow.id = workflowId;

            // Update existing workflow
            $.ajax({
                url: `/api/workflows/${workflowId}`,
                type: 'PUT',
                contentType: 'application/json',
                data: JSON.stringify(workflow),
                success: response => {
                    alert('Workflow saved successfully!');
                    // Update the UI with the saved workflow ID
                    $('#workflow-id').data('workflow-id', response.id);
                    $('#workflow-id').text(`ID: ${response.id}`);
                },
                error: error => {
                    alert('Error saving workflow: ' + error.responseText);
                }
            });
        } else {
            // Create new workflow
            $.ajax({
                url: '/api/workflows',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify(workflow),
                success: response => {
                    alert('Workflow created successfully!');
                    // Update the UI with the new workflow ID
                    $('#workflow-id').data('workflow-id', response.id);
                    $('#workflow-id').text(`ID: ${response.id}`);

                    // Redirect to the workflow editor for this workflow
                    window.location.href = `/workflow/${response.id}`;
                },
                error: error => {
                    alert('Error creating workflow: ' + error.responseText);
                }
            });
        }
    }

    /**
     * Export the current workflow
     * @returns {Object} Workflow data
     */
    exportWorkflow() {
        const workflowId = $('#workflow-id').data('workflow-id') || '';
        const workflowName = $('#workflow-name').val() || 'Exported Workflow';
        const processType = $('#workflow-process-type').val() || 'sequential';
        const maxIterations = parseInt($('#workflow-max-iterations').val()) || 10;

        // Get all nodes and edges
        const nodes = this.nodes.get();
        const edges = this.edges.get();

        // Create agent_ids array
        const agentIds = nodes.map(node => node.agentId);

        // Create tasks array
        const tasks = nodes.map(node => {
            const task = {
                id: 'task_' + node.id,
                agent_id: node.agentId,
                description: node.task ? node.task.description : '',
                expected_output: node.task ? node.task.expectedOutput : ''
            };

            // Add dependencies based on incoming edges
            const incomingEdges = edges.filter(edge => edge.to === node.id);
            if (incomingEdges.length > 0) {
                task.depends_on = incomingEdges.map(edge => {
                    const sourceNode = this.nodes.get(edge.from);
                    return sourceNode ? sourceNode.agentId : null;
                }).filter(id => id !== null);
            }

            return task;
        });

        // Create connections array
        const connections = edges.map(edge => {
            const sourceNode = this.nodes.get(edge.from);
            const targetNode = this.nodes.get(edge.to);

            if (!sourceNode || !targetNode) return null;

            return {
                source: sourceNode.agentId,
                target: targetNode.agentId,
                label: edge.label || '',
                type: edge.type || 'default',
                conditions: edge.condition ? { expression: edge.condition } : {}
            };
        }).filter(conn => conn !== null);

        // Create workflow object
        return {
            id: workflowId,
            name: workflowName,
            description: 'Anomaly detection workflow',
            process_type: processType,
            max_iterations: maxIterations,
            agent_ids: agentIds,
            tasks: tasks,
            connections: connections
        };
    }

    /**
     * Import a workflow
     * @param {Object} workflowData - Workflow data to import
     */
    importWorkflow(workflowData) {
        // Clear existing data
        this.nodes.clear();
        this.edges.clear();

        // Set workflow ID and name
        $('#workflow-id').data('workflow-id', '');
        $('#workflow-name').val(workflowData.name || 'Imported Workflow');
        $('#workflow-process-type').val(workflowData.process_type || 'sequential');
        $('#workflow-max-iterations').val(workflowData.max_iterations || 10);

        // Create a map of agents by ID for quick lookup
        const agentsById = {};
        this.agents.forEach(agent => {
            agentsById[agent.id] = agent;
        });

        // Create a map of tasks by agent ID
        const tasksByAgentId = {};
        if (workflowData.tasks && workflowData.tasks.length > 0) {
            workflowData.tasks.forEach(task => {
                tasksByAgentId[task.agent_id] = task;
            });
        }

        // Add agent nodes
        const nodeIdMap = {}; // Map agent IDs to node IDs

        if (workflowData.agent_ids && workflowData.agent_ids.length > 0) {
            // Position calculation
            const numAgents = workflowData.agent_ids.length;
            const radius = Math.min(this.container.clientWidth, this.container.clientHeight) * 0.4;
            const centerX = this.container.clientWidth / 2;
            const centerY = this.container.clientHeight / 2;

            workflowData.agent_ids.forEach((agentId, index) => {
                const agent = agentsById[agentId];
                if (agent) {
                    const angle = (2 * Math.PI * index) / numAgents;
                    const x = centerX + radius * Math.cos(angle);
                    const y = centerY + radius * Math.sin(angle);

                    // Get the task for this agent if it exists
                    const task = tasksByAgentId[agentId];

                    // Add the node
                    const nodeId = this.addAgentNode(agentId, x, y, task);
                    nodeIdMap[agentId] = nodeId;
                }
            });
        }

        // Add connections
        if (workflowData.connections && workflowData.connections.length > 0) {
            workflowData.connections.forEach(connection => {
                const fromNodeId = nodeIdMap[connection.source];
                const toNodeId = nodeIdMap[connection.target];

                if (fromNodeId && toNodeId) {
                    this.edges.add({
                        from: fromNodeId,
                        to: toNodeId,
                        label: connection.label || '',
                        type: connection.type || 'default',
                        condition: connection.conditions && connection.conditions.expression ? connection.conditions.expression : '',
                        arrows: {
                            to: { enabled: true, scaleFactor: 0.8 }
                        },
                        color: this._getEdgeColor(connection.type)
                    });
                }
            });
        } else if (workflowData.agent_ids && workflowData.agent_ids.length > 1) {
            // If no connections defined but multiple agents, create sequential connections
            for (let i = 0; i < workflowData.agent_ids.length - 1; i++) {
                const sourceId = nodeIdMap[workflowData.agent_ids[i]];
                const targetId = nodeIdMap[workflowData.agent_ids[i + 1]];

                if (sourceId && targetId) {
                    this.edges.add({
                        from: sourceId,
                        to: targetId,
                        arrows: {
                            to: { enabled: true, scaleFactor: 0.8 }
                        },
                        color: this._getEdgeColor('default')
                    });
                }
            }
        }

        // Fit the network to view all nodes
        this.fitToScreen();
    }

    /**
     * Delete the selected nodes and edges
     */
    deleteSelected() {
        const selectedNodes = this.network.getSelectedNodes();
        const selectedEdges = this.network.getSelectedEdges();

        if (selectedNodes.length > 0 || selectedEdges.length > 0) {
            if (confirm('Are you sure you want to delete the selected items?')) {
                this.nodes.remove(selectedNodes);
                this.edges.remove(selectedEdges);
            }
        }
    }

    /**
     * Zoom in on the network
     */
    zoomIn() {
        const currentScale = this.network.getScale();
        this.network.moveTo({ scale: currentScale * 1.2 });
    }

    /**
     * Zoom out on the network
     */
    zoomOut() {
        const currentScale = this.network.getScale();
        this.network.moveTo({ scale: currentScale * 0.8 });
    }

    /**
     * Fit the network to the screen
     */
    fitToScreen() {
        this.network.fit({ animation: true });
    }

    /**
     * Enable adding connections between nodes
     */
    enableConnectionMode() {
        this.network.addEdgeMode();
    }

    /**
     * Cancel current editing mode
     */
    disableEditMode() {
        this.network.disableEditMode();
    }

    /**
     * Create automatic layout for the nodes
     */
    autoLayout() {
        // Get all nodes and apply force-directed layout
        this.network.storePositions();

        // Reset the physics simulation
        this.network.setOptions({
            physics: {
                enabled: true,
                stabilization: {
                    enabled: true,
                    iterations: 100,
                    updateInterval: 10
                }
            }
        });

        // After stabilization is done, disable physics again
        this.network.on('stabilizationIterationsDone', () => {
            setTimeout(() => {
                this.network.setOptions({
                    physics: {
                        enabled: false
                    }
                });
                this.network.storePositions();
            }, 1000);
        });
    }
}
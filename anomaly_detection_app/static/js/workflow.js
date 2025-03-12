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

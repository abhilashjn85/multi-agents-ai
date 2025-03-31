import os
import json
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    session,
)
from werkzeug.utils import secure_filename
from controllers.workflow_controller import WorkflowController
from controllers.agent_controller import AgentController
from controllers.experiment_controller import ExperimentController
from controllers.results_controller import ResultsController
from models.agent import Agent
from models.workflow import Workflow
from models.experiment import Experiment
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Ensure upload directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_OUTPUT_FOLDER"], exist_ok=True)

# Initialize controllers
agent_controller = AgentController()
workflow_controller = WorkflowController()
experiment_controller = ExperimentController(app.config)
results_controller = ResultsController(app.config)

# Register dependencies between controllers
workflow_controller.register_agent_controller(agent_controller)
experiment_controller.register_controllers(agent_controller, workflow_controller)


@app.route("/")
def index():
    """Render the main dashboard"""
    agents = agent_controller.get_available_agents()
    workflows = workflow_controller.get_workflows()
    recent_experiments = experiment_controller.get_recent_experiments(5)
    return render_template(
        "index.html",
        agents=agents,
        workflows=workflows,
        recent_experiments=recent_experiments,
    )


@app.route("/workflow")
def workflow():
    """Render the workflow editor page"""
    agents = agent_controller.get_available_agents()
    workflows = workflow_controller.get_workflows()
    return render_template("workflow.html", agents=agents, workflows=workflows)


@app.route("/workflow/<workflow_id>")
def view_workflow(workflow_id):
    """View a specific workflow"""
    workflow = workflow_controller.get_workflow(workflow_id)
    agents = agent_controller.get_available_agents()
    return render_template("workflow.html", workflow=workflow, agents=agents)


@app.route("/api/workflows", methods=["GET", "POST"])
def api_workflows():
    """API endpoint for workflow CRUD operations"""
    if request.method == "GET":
        return jsonify(workflow_controller.get_workflows())
    elif request.method == "POST":
        data = request.json
        workflow = workflow_controller.create_workflow(data)
        return jsonify(workflow)


@app.route("/api/workflows/<workflow_id>", methods=["GET", "PUT", "DELETE"])
def api_workflow(workflow_id):
    """API endpoint for single workflow operations"""
    if request.method == "GET":
        return jsonify(workflow_controller.get_workflow(workflow_id))
    elif request.method == "PUT":
        data = request.json
        workflow = workflow_controller.update_workflow(workflow_id, data)
        return jsonify(workflow)
    elif request.method == "DELETE":
        workflow_controller.delete_workflow(workflow_id)
        return jsonify({"status": "success"})


@app.route("/api/agents", methods=["GET", "POST"])
def api_agents():
    """API endpoint to get all available agents or create new one"""
    if request.method == "GET":
        return jsonify(agent_controller.get_available_agents())
    elif request.method == "POST":
        data = request.json
        agent = agent_controller.create_agent(data)
        return jsonify(agent)


@app.route("/api/agents/<agent_id>", methods=["GET", "PUT", "DELETE"])
def api_agent(agent_id):
    """API endpoint for agent operations"""
    if request.method == "GET":
        return jsonify(agent_controller.get_agent(agent_id))
    elif request.method == "PUT":
        data = request.json
        agent = agent_controller.update_agent(agent_id, data)
        return jsonify(agent)
    elif request.method == "DELETE":
        agent_controller.delete_agent(agent_id)
        return jsonify({"status": "success"})


@app.route("/config")
def config():
    """Render the configuration page"""
    return render_template("config.html", config=experiment_controller.get_config())


@app.route("/api/config", methods=["GET", "PUT"])
def api_config():
    """API endpoint for configuration operations"""
    if request.method == "GET":
        return jsonify(experiment_controller.get_config())
    elif request.method == "PUT":
        data = request.json
        config = experiment_controller.update_config(data)
        return jsonify(config)


@app.route("/api/upload-config", methods=["POST"])
def api_upload_config():
    """API endpoint to upload a configuration file"""
    if "config_file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"})

    file = request.files["config_file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"})

    if file and allowed_file(file.filename, ["json"]):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Load the configuration into the experiment controller
        result = experiment_controller.load_config(file_path)

        return jsonify(
            {"status": "success", "message": "Configuration loaded successfully"}
        )

    return jsonify({"status": "error", "message": "Invalid file type"})


@app.route("/api/upload-data", methods=["POST"])
def api_upload_data():
    """API endpoint to upload a data file"""
    if "data_file" not in request.files:
        return jsonify({"status": "error", "message": "No file part"})

    file = request.files["data_file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"})

    if file and allowed_file(file.filename, ["csv", "xlsx"]):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Store the data path in the experiment controller
        experiment_controller.set_data_path(file_path)

        return jsonify(
            {
                "status": "success",
                "message": "Data file uploaded successfully",
                "path": file_path,
            }
        )

    return jsonify({"status": "error", "message": "Invalid file type"})


@app.route("/run", methods=["GET", "POST"])
def run_experiment():
    """Run an experiment with the current workflow and configuration"""
    if request.method == "POST":
        data = request.json
        workflow_id = data.get("workflow_id")
        config_id = data.get("config_id")

        # Create a new experiment
        experiment = experiment_controller.create_experiment(workflow_id, config_id)

        # Run the experiment in the background
        experiment_controller.run_experiment(experiment["id"])

        return jsonify({"status": "success", "experiment_id": experiment["id"]})

    # GET method - render the run experiment page
    workflows = workflow_controller.get_workflows()
    configs = experiment_controller.get_configs()
    return render_template("run.html", workflows=workflows, configs=configs)


@app.route("/api/run/<experiment_id>", methods=["GET"])
def api_experiment_status(experiment_id):
    """API endpoint to get experiment status"""
    status = experiment_controller.get_experiment_status(experiment_id)
    return jsonify(status)


@app.route("/api/run/<experiment_id>/cancel", methods=["POST"])
def api_cancel_experiment(experiment_id):
    """API endpoint to cancel an experiment"""
    result = experiment_controller.cancel_experiment(experiment_id)
    if result:
        return jsonify({"status": "success", "message": "Experiment cancelled"})
    return jsonify(
        {
            "status": "error",
            "message": "Failed to cancel experiment or experiment not found",
        }
    )


@app.route("/results/<experiment_id>")
def view_results(experiment_id):
    """View the results of an experiment"""
    experiment = experiment_controller.get_experiment(experiment_id)
    results = results_controller.get_results(experiment_id)
    return render_template("results.html", experiment=experiment, results=results)


@app.route("/api/results/<experiment_id>", methods=["GET"])
def api_results(experiment_id):
    """API endpoint to get experiment results"""
    results = results_controller.get_results(experiment_id)
    return jsonify(results)


def allowed_file(filename, allowed_extensions):
    """Check if the file has an allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)

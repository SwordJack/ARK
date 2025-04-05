#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   test_workflow.py
@Created:   2025/04/01 16:18
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from os import path
import pytest

from ark import LOGGER, file_system as fs
from ark.workflow import ExecutionEntity, WorkFlow, WorkUnit, GeneralWorkUnit
from ark.workflow import StatusCode

from .pseudo_api import UNIT_API_MAPPER

current_file_directory = path.dirname(__file__)
DATA_DIR = path.join(current_file_directory, "data")
WORK_DIR = path.join(current_file_directory, "work")

def test_workunit_checkpoint_producer(caplog):
    """A test for workunit."""
    fruit = "apple"
    unit_dict = {
        "api" : "checkpoint_producer",
        "store_as" : "producer",
        "args" : {
            "fruit_name" : fruit,
        }
    }
    workunit = WorkUnit.from_dict(unit_dict=unit_dict, debug=True)
    key, return_value = workunit.execute()
    assert key == "producer"
    assert fruit in return_value
    return

def test_workunit_self_inspection_unrecognized_api(caplog):
    """A test for unrecognized API."""
    unit_dict = {
        "api" : "missing_api",
        "store_as" : "missing",
        "args" : {
            "url" : "https://www.linux.org/",
        }
    }
    with pytest.raises(KeyError) as e:    # The KeyError is expected to be raised.
        workunit = WorkUnit.from_dict(unit_dict=unit_dict, debug=True)
        assert workunit.status == StatusCode.FAILED_INITIALIZATION
    assert '`missing_api` cannot be mapped' in caplog.text.lower()
    return

def test_workunit_self_inspection_missing_args(caplog):
    """A test for missing argument(s)."""
    unit_dict = {
        "api" : "checkpoint_producer",
        "store_as" : "producer",
        "args" : {
            "url" : "https://www.linux.org/",
        }
    }
    with pytest.raises(ValueError) as e:    # The ValueError is expected to be raised.
        workunit = WorkUnit.from_dict(unit_dict=unit_dict, debug=True)
        assert workunit.status == StatusCode.FAILED_INITIALIZATION
    assert 'missing required argument' in caplog.text.lower()
    return

def test_workflow_from_json_filepath(caplog):
    """Test initializing and executing Workflow from JSON filepath."""
    json_filepath = f'{DATA_DIR}/workflow_pseudo_loop.json'
    workflow = WorkFlow.from_json_filepath(json_filepath=json_filepath)
    workflow.execute()
    fs.remove_directory(WORK_DIR)
    # print(workflow.intermediate_data_mapper)
    assert "watermelon" in workflow.intermediate_data_mapper["producer_3"]
    assert hasattr(workflow.locate_by_identifier("loop@4"), "sub_workflows")      # The 5th workunit is a LoopWorkUnit.
    assert workflow.locate_by_identifier("workflow@4:loop_0").outer_data_mappers[-2] is workflow.intermediate_data_mapper    # The inner workflows' outer_mappers is cited from the intermediate_data_mapper of outer workflow.
    return

def test_general_from_json_filepath(caplog):
    """Test initializing and executing GeneralWorkunit from a json file."""
    json_filepath = f'{DATA_DIR}/general_pseudo_loop.json'
    general = GeneralWorkUnit.from_json_filepath(json_filepath=json_filepath, working_directory=WORK_DIR)
    assert "success" in caplog.text     # Indicate successful initialization.
    assert general.status == StatusCode.READY_TO_START
    general.execute()
    assert general.status == StatusCode.EXIT_WITH_ERROR_IN_INNER_UNITS
    assert general.locate_by_identifier("workflow@4:loop_0").status == StatusCode.EXIT_OK
    assert general.locate_by_identifier("workflow@4:loop_1").status == StatusCode.EXIT_WITH_ERROR_IN_INNER_UNITS
    assert general.locate_by_identifier("checkpoint_error@4:loop_1:0").status == StatusCode.EXIT_WITH_ERROR
    fs.remove_directory(WORK_DIR, empty_only=False)
    return

def test_general_reload_pass(caplog):
    """Test reloading a GeneralWorkUnit with a data mapper to cause updates."""
    json_filepath = f'{DATA_DIR}/general_pseudo_loop.json'
    general = GeneralWorkUnit.from_json_filepath(json_filepath=json_filepath, working_directory=WORK_DIR, save_snapshot=True)
    assert "success" in caplog.text     # Indicate successful initialization.
    general.execute()
    assert general.status == StatusCode.EXIT_WITH_ERROR_IN_INNER_UNITS
    assert general.locate_by_identifier("workflow@4:loop_0").status == StatusCode.EXIT_OK
    general.reload_json_filepath(json_filepath=json_filepath, data_mapper_for_reload={
        "fruit_1": "pear",
        "fruit_2": "apple",
    })
    assert general.status == StatusCode.SUSPECIOUS_UPDATES
    assert general.locate_by_identifier("loop@4").status == StatusCode.SUSPECIOUS_UPDATES
    assert not general.locate_by_identifier("checkpoint_error@4:loop_1:0")  # This workunit is erased during reloading since it is affected.
    general.execute()
    assert general.status == StatusCode.EXIT_WITH_ERROR_IN_INNER_UNITS
    assert general.locate_by_identifier("workflow@4:loop_0").status == StatusCode.EXIT_WITH_ERROR_IN_INNER_UNITS
    assert general.locate_by_identifier("workflow@4:loop_1").status == StatusCode.EXIT_OK
    assert general.locate_by_identifier("checkpoint_error@4:loop_1:0").status == StatusCode.EXIT_OK
    fs.remove_directory(WORK_DIR, empty_only=False)
    return

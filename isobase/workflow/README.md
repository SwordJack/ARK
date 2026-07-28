# Workflow

Author: [SwordJack](https://github.com/SwordJack)

[TOC]

## 1. Introduction

This module is to assemble JSON format task configuration files written in the specified format into a computing task, and supports saving the task as a binary file, re-reading the binary file, and reloading the task by reading the updated configuration file after the task exits with an error, as well as to continue execution from the updated/error-reporting position.

## 2. Quick Start

In this part, we will be provided with a step-by-step manual to learn the basic usage of this module, so as to be able to run our first computation job in 10 minutes.

### 2.1 Simple example

Before initializing your job, we need to prepare a json format configuration file as an _order_ (just like making an order of dishes you want to a restaurant).

Here's an example of a JSON configuration file (hereafter referred to as an **order**)

```json
{
  "api": "general",
  "store_as": "general",
  "args": {
    "fruit_1": "apple",
    "fruit_2": "orange",
    "fruit_3": "watermelon",
    "workunits": [
      {
        "api": "checkpoint_producer",
        "store_as": "producer_1",
        "args": {
          "fruit_name": "`fruit_1`"
        }
      },
      {
        "api": "checkpoint_producer",
        "store_as": "producer_2",
        "args": {
          "fruit_name": "`fruit_2`"
        }
      },
      {
        "api": "checkpoint_producer",
        "store_as": "producer_3",
        "args": {
          "fruit_name": "`fruit_3`"
        }
      },
      {
        "api": "union_values",
        "store_as": "unioned_values",
        "args": {
          "producer_1": "`producer_1`",
          "producer_2": "`producer_2`",
          "producer_3": "`producer_3`"
        }
      },
      {
        "api": "loop",
        "store_as": "loop_unit_lv0",
        "args": {
          "loop_data": "`unioned_values`",
          "loop_datum_varname": "unioned_value",
          "workunits": [
            {
              "api": "checkpoint_error",
              "store_as": "checkpoint_error",
              "args": {
                "test_str": "`unioned_value`"
              }
            },
            {
              "api": "print",
              "args": {
                "value": "`unioned_value`"
              }
            }
          ]
        }
      }
    ]
  }
}
```

In an order, each pair of large bracket (`{` and `}`) containing keys like `api`, `store_as` and `args` represents a workunit, and each pair of middle bracket (`[` and `]`) containing several workunits represents a workflow.

In each workunit,

- The value of the `api` key indicates the function called by the workunit and determines the type and function of the workunit;
- The value of the `store_as` key declares the name of the variable into which the return value of the function called will be stored;
- The value of the `args` key stores the values of the arguments that are to be used for the called function in the form of a dictionary.

In this order, the outermost level is a workunit with an `api` of `general`, which is a workunit for process control purposes, and it is only by using it as the outermost level that we can use the full features of this module.

The other `api` values we see, such as `checkpoint_producer`, `union_values`, `checkpoint_error`, etc., are specific Unit API values that are mapped to a callable object (function).

The way to read and run this order is as follows.

```python
from isobase.workflow import GeneralWorkUnit

# You can optionally import and pass a configured SqlDbService from the database module.
# from isobase.database import sql_db

general = GeneralWorkUnit.from_json_filepath(
    json_filepath="path/to/order.json",
    working_directory="your/working/directory",
    save_snapshot=True,
    overwrite_database=True
    # sql_service=sql_db  # Pass a unified DB service if you want PostgreSQL/MySQL. Defaults to local SQLite.
)
return_key, return_value = general.execute()
pickle_filepath = general.latest_pickle_filepath
```

When you created the `general` in the code above, a local SQLite database file is created in your working directory to update and track the execution status of your workflow. If your application already uses a relational database (like PostgreSQL) via the `isobase.database` module, you can inject it directly by passing `sql_service=sql_db`.

The data output throughout the execution of the `general` are able to be read from the `return_value`.

A snapshot is saved as a binary file to the `pickle_filepath` file at the end of execution.

- If the order we submit runs fine the first time we run it, we don't need to make any updates and just go ahead and read the data it outputs;
- If we face errors while running it, and you need to update your order but don't want to run it from scratch (because some of the steps are very time-consuming), you can use the following method to do so.

To commit any updates to your task, just make modifications to your order, and run following commands.

```python
# If you used a custom sql_service before, remember to inject it again during reload:
reloaded_general = GeneralWorkUnit.load_snapshot_file(filepath=pickle_filepath) # pass sql_service=sql_db if used
reloaded_general.reload(json_filepath="path/to/updated_order.json")
reloaded_general.execute()
```

Here, I'm using the `reloaded_general` variable to make it easier to distinguish between reloaded `general`, whereas you could actually just use the `general` variable and have the old object overwritten after executing `GeneralWorkUnit.load_snapshot_file`.

### 2.2 Assign values

When constructing a JSON workflow order, the root layer is a special control unit called `general`. The `args` dictionary of this `general` unit serves two critical purposes:

1. It defines the sequence of workunits via the `"workunits": [...]` key.
2. Every other key-value pair in this dictionary becomes a globally available variable!

For example:

```json
{
  "api": "general",
  "store_as": "general",
  "args": {
    "target_model": "gpt-4",
    "threshold": 0.8,
    "workunits": [ ... ]
  }
}
```

Any inner `workunit` can reference these values by wrapping the key in backticks: ``"`target_model`"`` and ``"`threshold`"``. This is the core mechanism of data mapping and inheritance in IsoBase.

#### `data_mapper_for_init`

If you are invoking a workflow programmatically in Python rather than from a purely static JSON file, there are times you want to pass non-JSON-serializable objects (like an existing database connection, a `pandas.DataFrame`, or a complex API client) into the workflow.

You can inject these variables using the `data_mapper_for_init` argument:

```python
complex_obj = MyCustomClient()
general = GeneralWorkUnit.from_json_filepath(
    json_filepath="order.json",
    data_mapper_for_init={"client_instance": complex_obj}
)
```

Inside `order.json`, you can then use ``"`client_instance`"`` to dynamically pass this object into your APIs.

---

### 2.3 Iterative Structures (Loop & Cluster Batch)

IsoBase workflow engine natively supports dynamic iteration over data collections without hardcoding parallel branches.

#### 1. Loop Structure (`LoopWorkUnit`)

A loop executes a sequence of inner units iteratively over an iterable object (like a list).

```json
{
  "api": "loop",
  "store_as": "loop_results",
  "args": {
    "loop_data": "`list_of_urls`",
    "loop_datum_varname": "current_url",
    "workunits": [
      {
        "api": "fetch_url",
        "args": { "url": "`current_url`" }
      }
    ]
  }
}
```

- **`loop_data`**: The target iterable. Must be evaluated to an iterable object (e.g., list, tuple) via data mapping.
- **`loop_datum_varname`**: The assigned variable name for the current item in the loop. Inside the `workunits` list, you reference the item using ``"`current_url`"``.
- **`run_in_subfolders` (Optional)**: If set to `true`, the workflow engine will create a sub-directory for each iteration (named after the iteration index) and execute the inner workunits within that isolated path. Excellent for parallel file generation tasks.

#### 2. Cluster Batch Structure (`ClusterBatchWorkUnit`, Under Construction)

If you are running the workflow on an HPC cluster environment (like Slurm), you can replace `"api": "loop"` with `"api": "cluster_batch"`.

The configuration syntax is nearly identical (`batch_data` instead of `loop_data`, and `batch_datum_varname` instead of `loop_datum_varname`). However, instead of executing the units sequentially on a single thread, the engine will:

1. Submit each iteration as a separate Slurm job script to the cluster queue.
2. Monitor the queue to respect the `max_simultaeneous_jobs` limit (default: 5).
3. Gather and union all output files seamlessly once the batch finishes.

This provides an immediate zero-code path to turn a sequential iteration into a massively parallel HPC batch job.

## 3. Advanced Manual

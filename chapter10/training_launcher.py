import argparse
import sys
import pip
import subprocess
import configparser
import json
import os
FLAGS = None

tf_config = json.loads(json.dumps({
    "cluster": {
        "worker": [],
        "ps": []
    },
   "task": {"type": "worker", "index": 0}
}))

def get_config(conf_file_path):
    config = configparser.ConfigParser()
    config.read(conf_file_path)
    return config

def get_cluster_components(config):
    chief = None
    if "chief" in config["cluster"].keys():
        chief = config["cluster"]["chief"]
        chief = [x.strip() for x in chief.split(',')]
    workers = None
    if "worker" in config["cluster"].keys():
        workers= config["cluster"]["worker"]
        workers = [x.strip() for x in workers.split(',')]

    ps = None
    if "ps" in config["cluster"].keys():
        ps= config["cluster"]["ps"]
        ps = [x.strip() for x in ps.split(',')]

    return chief,workers,ps


def init_tf_config(master, workers, ps):
    if master !=None:
        tf_config["cluster"]["chief"] = master
    if workers != None:
        tf_config["cluster"]["worker"] = workers
    if ps !=None:
        tf_config["cluster"]["ps"] = ps


def get_tf_config(role, index):
    tf_config["task"]["type"] =role
    tf_config["task"]["index"] = index

def get_launch_command(py_script, ssh_host):
    env_variable = str(json.dumps(tf_config)).replace("\"", "\\\"")
    command1 = "python "+py_script+" --input_path "+FLAGS.input_path+" --output_path "+FLAGS.output_path
    command2 = "\"export TF_CONFIG='" + env_variable + "';" + command1 + "\""
    command = command2 + " & "
    if ssh_host != "localhost":
        command = "ssh "+ssh_host +" "+ command2+ " &"
    return command


def train():
    config = get_config(FLAGS.cluster_conf_path)
    master,workers,ps = get_cluster_components(config)
    init_tf_config(master,workers,ps)
    counter = 0
    if master is not None:
        for host in master:
            parts = host.split(":")
            h = parts[0]
            p = parts[1]
            get_tf_config("chief", counter)
            cmd = get_launch_command(FLAGS.python_script_path, h)
            print(cmd)
            os.system(cmd)
            counter = counter+1
    counter = 0
    if workers is not None:
        for host in workers:
            parts = host.split(":")
            h = parts[0]
            p = parts[1]
            get_tf_config("worker", counter)
            cmd = get_launch_command(FLAGS.python_script_path, h)
            print(cmd)
            os.system(cmd)
            counter = counter+1
    counter = 0
    if ps is not None:
        for host in ps:
            parts = host.split(":")
            h = parts[0]
            p = parts[1]
            get_tf_config("ps", counter)
            cmd = get_launch_command(FLAGS.python_script_path, h)
            print(cmd)
            os.system(cmd)
            counter = counter + 1


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--python_script_path",
      type=str,
      default="",
      help="Path of python file that has the model training code"
  )

  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--input_path",
      type=str,
      default="",
      help="Directory path to the input file. Could you be clous storage"
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default="",
      help="Directory path to the input file. Could you be clous storage"
  )
  parser.add_argument(
      "--cluster_conf_path",
      type=str,
      default="cluster.conf",
      help="Path to conf file containing cluster host list"
  )
  FLAGS, unparsed = parser.parse_known_args()
  train()
  print("Model training started.")





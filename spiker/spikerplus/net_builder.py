import re
import json
import logging
import torch.nn as nn
from typing import Dict, List, Any


class SNN(nn.Module):
    def __init__(self, net_dict: Dict[str, Any]):

        self.default_dict: Dict[str, Any] = {
            "n_cycles": 73,
            "n_inputs": 40,
            "layer_0": {
                "neuron_model": "lif",
                "n_neurons": 128,
                "alpha": 0.9,
                "learn_alpha": False,
                "beta": 0.9375,
                "learn_beta": False,
                "threshold": 1.0,
                "learn_threshold": False,
                "reset_mechanism": "subtract",
            },
            "layer_1": {
                "neuron_model": "lif",
                "n_neurons": 10,
                "alpha": 0.9,
                "learn_alpha": False,
                "beta": 0.9375,
                "learn_beta": False,
                "threshold": 1.0,
                "learn_threshold": False,
                "reset_mechanism": "none",
            },
        }

        self.net_allowed_keys: List[str] = self.select_keys()
        self.supported_models: List[str] = ["if", "lif", "syn", "rif", "rlif", "rsyn"]

        self.has_alpha: Dict[str, bool] = {
            "if": False,
            "lif": False,
            "syn": True,
            "rif": False,
            "rlif": False,
            "rsyn": True,
        }

        self.has_beta: Dict[str, bool] = {
            "if": False,
            "lif": True,
            "syn": True,
            "rif": False,
            "rlif": True,
            "rsyn": True,
        }

        self.supported_resets: List[str] = ["zero", "subtract", "none"]

        self.net_dict: Dict[str, Any] = self.parse_config(net_dict)

    def build(self):

        snn = SNN(self.net_dict)

        log_message = "Network ready: " + str(snn) + "\n"
        logging.info(log_message)

        return snn

    def select_keys(self) -> List[str]:

        keywords = self.default_dict.keys()

        allowed_keys: List[str] = []

        for k in keywords:
            k = re.sub(r"\d+", "", k)

            if k not in allowed_keys:
                allowed_keys.append(k)

        return allowed_keys

    def parse_config(self, net_dict: Dict[str, Any]) -> Dict[str, Any]:

        parsed_dict: Dict[str, Any] = {}

        for key in net_dict:
            if any([allowed in key for allowed in self.net_allowed_keys]):
                if "layer" not in key:
                    if type(net_dict[key]) is not int:
                        raise ValueError(
                            "Error, " + key + " must be an integer value\n"
                        )

                    parsed_dict[key] = net_dict[key]

                else:
                    parsed_dict[key] = {}
                    layer: Dict[str, Any] = net_dict[key]

                    if "n_neurons" in layer.keys():
                        if not isinstance(layer["n_neurons"], int):
                            raise ValueError("Number of neurons must be integer.\n")

                        else:
                            parsed_dict[key]["n_neurons"] = layer["n_neurons"]

                    else:
                        parsed_dict[key]["n_neurons"] = self.default_dict["layer_0"][
                            "n_neurons"
                        ]

                    if "neuron_model" in layer.keys():
                        if layer["neuron_model"] not in self.supported_models:
                            raise ValueError(
                                "Unsupported neuron model. "
                                "Choose one between "
                                + str(self.supported_models)
                                + "\n"
                            )

                        else:
                            parsed_dict[key]["neuron_model"] = layer["neuron_model"]

                    else:
                        parsed_dict[key]["neuron_model"] = self.default_dict["layer_0"][
                            "neuron_model"
                        ]

                    if "threshold" in layer.keys():
                        if not isinstance(layer["threshold"], (int, float)):
                            raise ValueError("Treshold must be numeric.\n")

                        else:
                            parsed_dict[key]["threshold"] = layer["threshold"]

                    else:
                        parsed_dict[key]["threshold"] = self.default_dict["layer_0"][
                            "threshold"
                        ]

                    if "learn_threshold" in layer.keys():
                        if not isinstance(layer["learn_threshold"], bool):
                            raise ValueError("learn_threshold must be boolean.\n")

                        else:
                            parsed_dict[key]["learn_threshold"] = layer[
                                "learn_threshold"
                            ]

                    else:
                        parsed_dict[key]["learn_threshold"] = self.default_dict[
                            "layer_0"
                        ]["learn_threshold"]

                    if "reset_mechanism" in layer.keys():
                        if layer["reset_mechanism"] not in self.supported_resets:
                            raise ValueError(
                                "Invalid reset mechanism. "
                                "Choose one between "
                                + str(self.supported_resets)
                                + "\n"
                            )

                        else:
                            parsed_dict[key]["reset_mechanism"] = layer[
                                "reset_mechanism"
                            ]

                    else:
                        parsed_dict[key]["reset_mechanism"] = self.default_dict[
                            "layer_0"
                        ]["reset_mechanism"]

                    if self.has_alpha[parsed_dict[key]["neuron_model"]]:
                        if "alpha" in layer.keys():
                            if not isinstance(layer["alpha"], float):
                                raise ValueError("Alpha decay must be float\n")

                            elif layer["alpha"] < 0.0 or layer["alpha"] > 1.0:
                                raise ValueError(
                                    "Alpha decay must be between 0 and 1\n"
                                )

                            else:
                                parsed_dict[key]["alpha"] = layer["alpha"]

                        else:
                            parsed_dict[key]["alpha"] = self.default_dict["layer_0"][
                                "alpha"
                            ]

                        if "learn_alpha" in layer.keys():
                            if not isinstance(layer["learn_alpha"], bool):
                                raise ValueError("learn_alpha must be boolean.\n")

                            else:
                                parsed_dict[key]["learn_alpha"] = layer["learn_alpha"]

                        else:
                            parsed_dict[key]["learn_alpha"] = self.default_dict[
                                "layer_0"
                            ]["learn_alpha"]

                    if self.has_beta[parsed_dict[key]["neuron_model"]]:
                        if "beta" in layer.keys():
                            if not isinstance(layer["beta"], float):
                                raise ValueError("Beta decay must be float\n")

                            elif layer["beta"] < 0.0 or layer["beta"] > 1.0:
                                raise ValueError("Beta decay must be between 0 and 1\n")

                            else:
                                parsed_dict[key]["beta"] = layer["beta"]

                        else:
                            parsed_dict[key]["beta"] = self.default_dict["layer_0"][
                                "beta"
                            ]

                        if "learn_beta" in layer.keys():
                            if not isinstance(layer["learn_beta"], bool):
                                raise ValueError("learn_beta must be boolean.\n")

                            else:
                                parsed_dict[key]["learn_beta"] = layer["learn_beta"]

                        else:
                            parsed_dict[key]["learn_beta"] = self.default_dict[
                                "layer_0"
                            ]["learn_beta"]

        if "n_cycles" not in parsed_dict:
            parsed_dict["n_cycles"] = self.default_dict["n_cycles"]

        if "n_inputs" not in parsed_dict:
            parsed_dict["n_inputs"] = self.default_dict["n_inputs"]

        at_least_one_layer: bool = False
        for key in parsed_dict:
            if "layer_" in key:
                at_least_one_layer = True

        if not at_least_one_layer:
            for key in self.default_dict:
                if "layer_" in key:
                    parsed_dict[key] = self.default_dict[key]

        log_message: str = "Network configured: \n"
        log_message += json.dumps(parsed_dict, indent=4) + "\n"

        logging.info(log_message)

        return parsed_dict

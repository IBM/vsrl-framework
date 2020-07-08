#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

# todo refactor this file to work on a generic machine (or at least a Docker container).
import logging
import subprocess
import urllib.request
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Dict, Optional

import vsrl.verifier.expr as expr
import vsrl.verifier.expr_helpers as expr_helpers
from vsrl.verifier.keymaerax_pretty_printer import keymaerax_pp
from vsrl.verifier.monitor import Monitor

# TODO move this into the utils package and infer automatically.
PATHS = {
    "KeYmaera X": [
        "/tmp/keymaerax.jar",
        str(Path.home() / "keymaerax.jar"),
        "keymaerax.jar",
    ],
    "MathKernel": [
        "/usr/local/Wolfram/WolframEngine/12.0/Executables/MathKernel",
        "/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel",
    ],
    "Wolfram JLink": [
        "/usr/local/Wolfram/WolframEngine/12.0/SystemFiles/Links/JLink/SystemFiles/Libraries/Linux-x86-64",
        "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/SystemFiles/Links/JLink/SystemFiles/Libraries/MacOSX-x86-64",
    ],
    "java": ["/Users/nathan@ibm.com/.jenv/shims/java"],
}


class KeYmaeraXTool(Monitor):
    """
    An interface to the KeYmaera X Theorem Prover for Hybrid Systems
    """

    def ctrl_action_is_safe(
        self, state: Dict[expr.Variable, expr.Number], action
    ) -> bool:
        pass

    def model_is_accurate(
        self,
        state: Dict[expr.Variable, expr.Number],
        action,
        next_state: Dict[expr.Variable, expr.Number],
    ) -> bool:
        pass

    # todo implement
    def model_monitor(self):
        pass

    # todo implement
    def controller_monitor(self):
        pass

    # This flag is set to true in the constructor.
    INSTALL_CHECKED = False

    # If set to True, all .kyx and .kyt and .kyp files will be kept on the file system.
    KYX_BRIDGE_DEBUGGING = False

    def __init__(self, java_command=None, jar_location=None):
        self.jar_location = jar_location or KeYmaeraXTool._infer_jar_location()
        self.java_command = java_command or KeYmaeraXTool._infer_java_command()
        if not KeYmaeraXTool.INSTALL_CHECKED:
            logging.debug("Configuring KeYmaera X.")
            assert self.is_configured(), "KeYmaera X is not properly configured."
            KeYmaeraXTool.INSTALL_CHECKED = True
            logging.info("checking KeYmaera X installation.")

    @staticmethod
    def _infer_java_command():
        # todo generalize this to use jenv if it's availalbe and otherwise use java.
        # or, better, just check that java 1.8 is installed.
        return KeYmaeraXTool._get_location("java", log_error=False) or "java"

    @staticmethod
    def _infer_jar_location():
        kx_location = KeYmaeraXTool._get_location("KeYmaera X")
        if kx_location:
            logging.info(f"Found KeYmaera X installation at: {kx_location}")
            return kx_location
        else:
            # Could not find locally. Try to download.
            logging.warn("Trying to download KeYmaera X jar file.")
            try:
                download_location = "keymaerax.jar"
                response = urllib.request.urlopen("http://keymaerax.org/keymaerax.jar")
                with open(download_location, "wb") as f:
                    f.write(response.read())
                return download_location
            except Exception as e:
                logging.error(e)
                logging.error(
                    "Could not file KeYmaera X locally and could not download from keymaerax.org",
                )
                return None

    def is_configured(self):
        return self.check(
            expr.Eq(expr.Number(2), expr.Plus(expr.Number(1), expr.Number(1))), "QE"
        )

    @staticmethod
    def _get_location(file: str, log_error: bool = True) -> Optional[str]:
        """
        Returns the first existing file from `PATHS[file]`.
        :param log_error: if True, an error is logged if no path for `file` is found.
        """
        for candidate in PATHS[file]:
            if Path(candidate).is_file():
                return candidate
        # TODO - should this be an error or a warning?
        if log_error:
            logging.error(f"Could not find {file}.")

    def _check_with_strs(self, formula: expr.Formula, belle_tactic: str) -> bool:
        """
        Checks whether the tactic string `belle_tactic` proves the formula `formula`.
        :param formula:
        :param belle_tactic:
        :return: Boolean value -- true if the formula proved the formula and false otherwise.
        """
        assert isinstance(formula, expr.Formula) and isinstance(belle_tactic, str)

        model_file_contents = self._formula_to_file(formula)
        model_file_name = self._write_model_file(model_file_contents)
        proof_file_name = self._write_proof_file(belle_tactic)

        mathkernel_location = KeYmaeraXTool._get_location("MathKernel")
        jlink_location = KeYmaeraXTool._get_location("Wolfram JLink")
        if mathkernel_location is not None and jlink_location is not None:
            wolframscript_options = (
                f"-mathkernel '{mathkernel_location}' -jlink '{jlink_location}'"
            )
        else:
            wolframscript_options = ""

        command = "%s -Xss20M -jar '%s'  -launch  -prove %s -tactic %s %s" % (
            self.java_command,
            self.jar_location,
            model_file_name,
            proof_file_name,
            wolframscript_options,
        )
        logging.info("KeYmaera X is running command:\n\t%s" % command)
        if self.KYX_BRIDGE_DEBUGGING:
            logging.info(command)

        output = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = output.communicate()

        success = False
        for line in out.decode("utf-8").splitlines():
            if line.startswith("PROVED"):
                success = True
                if self.KYX_BRIDGE_DEBUGGING:
                    logging.info(
                        "Found a line that indicates the proof is complete: %s" % line
                    )
                    logging.info("Finished proof of: %s" % keymaerax_pp(formula))
                break
            else:
                logging.info(
                    "The following line does NOT indicate that the proof was completed: %s"
                    % line
                )

        if not self.KYX_BRIDGE_DEBUGGING:
            Path(model_file_name).unlink()
            Path(proof_file_name).unlink()
            proof_file = Path(model_file_name + ".kyp")
            if (proof_file).is_file():
                proof_file.unlink()
            else:
                logging.warn(
                    "Could not find the proof file - may be leaving a mess on the file system.",
                )
        else:
            logging.info(
                "Temporary files are NOT being deleted because KeYmaeraX.KYX_BRIDGE_DEBUGGING=True",
            )

        if success:
            logging.info(
                "KyX Bridge just proved %s with %s\n\tin %s\n\toutput: %s"
                % (keymaerax_pp(formula), belle_tactic, model_file_name, out),
            )
        elif self.KYX_BRIDGE_DEBUGGING:
            logging.info(
                "KyX Bridge DID NOT prove %s with %s\n\tin %s\n\toutput: %s"
                % (
                    keymaerax_pp(formula),
                    belle_tactic,
                    model_file_name,
                    out.decode("utf-8"),
                ),
            )
        return success

    def check(self, formula: expr.Formula, belle_tactic: str) -> bool:
        """

        :param formula:
        :param belle_tactic:
        :return: boolean -- true if KeYmaera X succeeds in proving the formula using the tactic.
        """
        assert isinstance(
            formula, expr.Formula
        ), f"expected formula but found {formula}"
        return self._check_with_strs(formula, belle_tactic)

    def check_arithmetic(self, formula: expr.Formula) -> bool:
        return self.check(formula, "master")

    @staticmethod
    def _formula_to_file(formula: expr.Formula) -> str:
        assert isinstance(formula, expr.Formula)
        variables = expr_helpers.variables(formula)
        variable_str = "\n".join(f"Real {keymaerax_pp(v)};" for v in variables)

        file_contents = f"""\
        ProgramVariables
        {variable_str}
        End.

        Problem
        {keymaerax_pp(formula)}
        End.
        """

        return dedent(file_contents)

    @staticmethod
    def _write_model_file(model):
        fn = f"{uuid.uuid4().int}.kyx"
        Path(fn).write_bytes(model.encode("utf8"))
        return fn

    @staticmethod
    def _write_proof_file(proof):
        fn = f"{uuid.uuid4().int}.kyt"
        Path(fn).write_bytes(proof.encode("utf8"))
        return fn

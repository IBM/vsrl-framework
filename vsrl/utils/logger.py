#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

"""
Comet
-----

Mainly taken from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/comet.py
(until https://github.com/PyTorchLightning/pytorch-lightning/issues/2393 is resolved)
See license for the original code at the bottom of this file.
"""

from argparse import Namespace
from pathlib import Path
from typing import Optional, Dict, Union, Any

try:
    from comet_ml import Experiment as CometExperiment
    from comet_ml import ExistingExperiment as CometExistingExperiment
    from comet_ml import OfflineExperiment as CometOfflineExperiment
    from comet_ml import BaseExperiment as CometBaseExperiment
    try:
        from comet_ml.api import API
    except ImportError:  # pragma: no-cover
        # For more information, see: https://www.comet.ml/docs/python-sdk/releases/#release-300
        from comet_ml.papi import API  # pragma: no-cover

    _COMET_AVAILABLE = True
except ImportError:  # pragma: no-cover
    CometExperiment = None
    CometExistingExperiment = None
    CometOfflineExperiment = None
    CometBaseExperiment = None
    API = None
    _COMET_AVAILABLE = False


import torch
from torch import is_tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_only


class CometLogger(LightningLoggerBase):
    r"""
    Log using `Comet.ml <https://www.comet.ml>`_. Install it with pip:
    .. code-block:: bash
        pip install comet-ml
    Comet requires either an API Key (online mode) or a local directory path (offline mode).
    **ONLINE MODE**
    Example:
        >>> import os
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CometLogger
        >>> # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        >>> comet_logger = CometLogger(
        ...     api_key=os.environ.get('COMET_API_KEY'),
        ...     workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
        ...     save_dir='.',  # Optional
        ...     project_name='default_project',  # Optional
        ...     rest_api_key=os.environ.get('COMET_REST_API_KEY'),  # Optional
        ...     experiment_name='default'  # Optional
        ... )
        >>> trainer = Trainer(logger=comet_logger)
    **OFFLINE MODE**
    Example:
        >>> from pytorch_lightning.loggers import CometLogger
        >>> # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        >>> comet_logger = CometLogger(
        ...     save_dir='.',
        ...     workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
        ...     project_name='default_project',  # Optional
        ...     rest_api_key=os.environ.get('COMET_REST_API_KEY'),  # Optional
        ...     experiment_name='default'  # Optional
        ... )
        >>> trainer = Trainer(logger=comet_logger)
    Args:
        api_key: Required in online mode. API key, found on Comet.ml
        save_dir: Required in offline mode. The path for the directory to save local comet logs
        workspace: Optional. Name of workspace for this user
        project_name: Optional. Send your experiment to a specific project.
            Otherwise will be sent to Uncategorized Experiments.
            If the project name does not already exist, Comet.ml will create a new project.
        rest_api_key: Optional. Rest API key found in Comet.ml settings.
            This is used to determine version number
        experiment_name: Optional. String representing the name for this particular experiment on Comet.ml.
        experiment_key: Optional. If set, restores from existing experiment.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 workspace: Optional[str] = None,
                 project_name: Optional[str] = None,
                 rest_api_key: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 experiment_key: Optional[str] = None,
                 force_offline: bool = False,
                 **kwargs):

        if not _COMET_AVAILABLE:
            raise ImportError('You want to use `comet_ml` logger which is not installed yet,'
                              ' install it with `pip install comet-ml`.')
        super().__init__()
        self._experiment = None
        self.save_dir = save_dir
        self.api_key = api_key

        # Determine online or offline mode based on which arguments were passed to CometLogger
        if api_key is None and save_dir is None:
            # If neither api_key nor save_dir are passed as arguments, raise an exception
            raise MisconfigurationException("CometLogger requires either api_key or save_dir during initialization.")
        elif api_key is None:
            self.mode = "offline"
        elif save_dir is None:
            self.mode = "online"
        else: # both given so need explicit argument
            self.mode = "offline" if force_offline else "online"

        log.info(f"CometLogger will be initialized in {self.mode} mode")

        self.workspace = workspace
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.experiment_key = experiment_key
        self._kwargs = kwargs

        if rest_api_key is not None:
            # Comet.ml rest API, used to determine version number
            self.rest_api_key = rest_api_key
            self.comet_api = API(self.rest_api_key)
        else:
            self.rest_api_key = None
            self.comet_api = None

        if self.experiment_name is None:
            self._version = self.experiment.id
        else:
            # ensure that the directory name is unique by appending a number to the end
            root_save_dir = Path(self.save_dir) / self.name
            root_save_dir.mkdir(exist_ok=True, parents=True)
            max_i = -1
            for path in root_save_dir.glob("*"):
                name = path.name
                if name.startswith(self.experiment_name):
                    try:
                        i = int(name.split("_")[-1])
                        max_i = max(max_i, i)
                    except (IndexError, ValueError): # no _ or no int at end
                        continue
            self._version = f"{self.experiment_name}_{max_i + 1}"

        self._kwargs = kwargs

    @property
    def experiment(self) -> CometBaseExperiment:
        r"""
        Actual Comet object. To use Comet features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            self.logger.experiment.some_comet_function()
        """
        if self._experiment is not None:
            return self._experiment

        if self.mode == "online":
            if self.experiment_key is None:
                self._experiment = CometExperiment(
                    api_key=self.api_key,
                    workspace=self.workspace,
                    project_name=self.project_name,
                    **self._kwargs
                )
                self.experiment_key = self._experiment.get_key()
            else:
                self._experiment = CometExistingExperiment(
                    api_key=self.api_key,
                    workspace=self.workspace,
                    project_name=self.project_name,
                    previous_experiment=self.experiment_key,
                    **self._kwargs
                )
        else:
            save_dir = Path(self.save_dir) / self.name / self.version
            save_dir.mkdir(exist_ok=True, parents=True)
            self._experiment = CometOfflineExperiment(
                offline_directory=save_dir,
                workspace=self.workspace,
                project_name=self.project_name,
                **self._kwargs
            )

        if self.experiment_name is not None:
            self._experiment.set_name(self.experiment_name)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        self.experiment.log_parameters(params)

    @rank_zero_only
    def log_metrics(
            self,
            metrics: Dict[str, Union[torch.Tensor, float]],
            step: Optional[int] = None
    ) -> None:
        # Comet.ml expects metrics to be a dictionary of detached tensors on CPU
        for key, val in metrics.items():
            if is_tensor(val):
                metrics[key] = val.cpu().detach()

        self.experiment.log_metrics(metrics, step=step)

    def reset_experiment(self):
        self._experiment = None

    @rank_zero_only
    def finalize(self, status: str) -> None:
        r"""
        When calling ``self.experiment.end()``, that experiment won't log any more data to Comet.
        That's why, if you need to log any more data, you need to create an ExistingCometExperiment.
        For example, to log data when testing your model after training, because when training is
        finalized :meth:`CometLogger.finalize` is called.
        This happens automatically in the :meth:`~CometLogger.experiment` property, when
        ``self._experiment`` is set to ``None``, i.e. ``self.reset_experiment()``.
        """
        self.experiment.end()
        self.reset_experiment()

    @property
    def name(self) -> str:
        return str(self.project_name)

    # no setter because you can't change the project name of an experiment (I don't think)

    @property
    def version(self) -> str:
        return self._version

"""
Summary of modifications:
* the name property is now consistently the project name and can't be altered
* the version is the experiment name if given otherwise the id
* the save directory for offline experiments is save_dir / name / version
* added explicit argument for online vs offline logging

License for the original version of this code:
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2018-2020 William Falcon

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   """

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ADB helper for QNN execution. Foundation-internal."""

from __future__ import annotations

import fnmatch
import os
import subprocess


class ADBRunner:
    """adb push/shell/pull for QNN device execution."""

    DEVICE_WORKSPACE = "/data/local/tmp/foundation_runner"

    def __init__(self, build_path: str, device_id: str, qnn_sdk: str, soc_model: str):
        self.build_path = build_path
        self.device_id = device_id
        self.qnn_sdk = qnn_sdk
        self.soc_model = soc_model
        self.target = "aarch64-android"
        self._htp_arch = self._get_htp_arch()

    def _get_htp_arch(self) -> str:
        """SoC → HTP arch 버전."""
        mapping = {
            "SM8750": "79",
            "SM8650": "75",
            "SM8550": "73",
            "SM8450": "69",
            "SM8350": "68",
        }
        return mapping.get(self.soc_model, "73")

    def _adb(self, cmd: list, capture: bool = False) -> str:
        full = ["adb", "-s", self.device_id] + cmd
        if capture:
            r = subprocess.run(full, capture_output=True, text=True)
            return r.stdout + r.stderr
        subprocess.run(full, check=False)
        return ""

    def setup_workspace(self) -> None:
        self._adb(["shell", f"rm -rf {self.DEVICE_WORKSPACE}"])
        self._adb(["shell", f"mkdir -p {self.DEVICE_WORKSPACE}"])

    def push_qnn_libs(self) -> None:
        """QNN .so 라이브러리 푸시."""
        arch = self._htp_arch
        libs = [
            f"{self.qnn_sdk}/lib/{self.target}/libQnnHtp.so",
            f"{self.qnn_sdk}/lib/hexagon-v{arch}/unsigned/libQnnHtpV{arch}Skel.so",
            f"{self.qnn_sdk}/lib/{self.target}/libQnnHtpV{arch}Stub.so",
            f"{self.qnn_sdk}/lib/{self.target}/libQnnHtpPrepare.so",
            f"{self.qnn_sdk}/lib/{self.target}/libQnnSystem.so",
            f"{self.build_path}/backends/qualcomm/libqnn_executorch_backend.so",
            f"{self.qnn_sdk}/lib/{self.target}/libQnnModelDlc.so",
        ]
        for lib in libs:
            if os.path.exists(lib):
                self._adb(["push", lib, self.DEVICE_WORKSPACE])
            else:
                print(f"[foundation] 경고: 라이브러리 없음 {lib}")

    def push_file(self, local_path: str) -> None:
        self._adb(["push", local_path, self.DEVICE_WORKSPACE])

    def push_dir_files(self, local_dir: str, glob: str = "*.bin") -> None:
        """로컬 디렉토리의 파일들을 디바이스로 푸시."""
        for fname in sorted(os.listdir(local_dir)):
            if fnmatch.fnmatch(fname, glob):
                self._adb(["push", os.path.join(local_dir, fname), self.DEVICE_WORKSPACE])

    def execute(self, cmd: str) -> str:
        """adb shell 에서 커맨드 실행 후 stdout 반환."""
        full_cmd = f"cd {self.DEVICE_WORKSPACE} && {cmd}"
        return self._adb(["shell", full_cmd], capture=True)

    def pull(self, device_path: str, local_path: str) -> None:
        self._adb(["pull", device_path, local_path])

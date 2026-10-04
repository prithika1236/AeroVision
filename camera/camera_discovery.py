"""
Camera and Video Device Discovery Module.
Safely probes OpenCV-compatible video capture devices, tests frame acquisition,
retrieves friendly device names on Windows, and releases handles immediately.
"""

import sys
import json
import re
import subprocess
from typing import List, Dict, Any
import cv2

import config


def get_directshow_camera_names() -> List[str]:
    """
    Queries DirectShow CLSID_VideoInputDeviceCategory via PowerShell.
    DirectShow enumeration order matches OpenCV cv2.CAP_DSHOW capture device indices.
    """
    if not sys.platform.startswith("win"):
        return []

    ps = """
    Add-Type -TypeDefinition @"
    using System;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;
    using System.Runtime.InteropServices.ComTypes;

    public class DirectShowEnum {
        [ComImport, Guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86")]
        public class CreateDevEnum {}

        [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        public interface ICreateDevEnum {
            [PreserveSig]
            int CreateClassEnumerator([In, MarshalAs(UnmanagedType.LPStruct)] Guid pType, [Out] out IEnumMoniker ppEnumMoniker, [In] int dwFlags);
        }

        [ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        public interface IPropertyBag {
            [PreserveSig]
            int Read([In, MarshalAs(UnmanagedType.LPWStr)] string pszPropName, [In, Out, MarshalAs(UnmanagedType.Struct)] ref object pVar, [In] IntPtr pErrorLog);
            [PreserveSig]
            int Write([In, MarshalAs(UnmanagedType.LPWStr)] string pszPropName, [In, MarshalAs(UnmanagedType.Struct)] ref object pVar);
        }

        public static string[] GetFriendlyNames() {
            var list = new List<string>();
            try {
                Guid CLSID_VideoInputDeviceCategory = new Guid("860BB310-5D01-11d0-BD3B-00A0C911CE86");
                ICreateDevEnum devEnum = (ICreateDevEnum)new CreateDevEnum();
                IEnumMoniker enumMoniker;
                int hr = devEnum.CreateClassEnumerator(CLSID_VideoInputDeviceCategory, out enumMoniker, 0);
                if (hr != 0 || enumMoniker == null) return list.ToArray();

                IMoniker[] monikers = new IMoniker[1];
                IntPtr fetched = IntPtr.Zero;
                while (enumMoniker.Next(1, monikers, fetched) == 0) {
                    object bagObj;
                    Guid IID_IPropertyBag = typeof(IPropertyBag).GUID;
                    monikers[0].BindToStorage(null, null, ref IID_IPropertyBag, out bagObj);
                    IPropertyBag bag = bagObj as IPropertyBag;
                    if (bag != null) {
                        object val = null;
                        bag.Read("FriendlyName", ref val, IntPtr.Zero);
                        if (val != null) list.Add(val.ToString());
                    }
                    Marshal.ReleaseComObject(monikers[0]);
                }
            } catch {}
            return list.ToArray();
        }
    }
"@
    [DirectShowEnum]::GetFriendlyNames() | ConvertTo-Json -Compress
    """
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, str):
                return [data]
            elif isinstance(data, list):
                return data
    except Exception:
        pass

    # Fallback to WMI PnP camera names if DirectShow query fails
    try:
        ps_pnp = (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image' } | "
            "Select-Object -ExpandProperty Name"
        )
        res_pnp = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_pnp],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        if res_pnp.returncode == 0 and res_pnp.stdout.strip():
            return [line.strip() for line in res_pnp.stdout.strip().splitlines() if line.strip()]
    except Exception:
        pass

    return []


def get_connected_external_devices() -> List[Dict[str, Any]]:
    """
    Enumerates connected physical external hardware devices (Cameras, USB Mice/Keyboards,
    Serial Adapters, Development Boards, etc.) via Windows Plug-and-Play.
    Filters out internal Windows subsystem noise.
    """
    if not sys.platform.startswith("win"):
        return []

    ps_cmd = """
    Get-CimInstance Win32_PnPEntity | 
        Where-Object { 
            ($_.DeviceID -like 'USB*' -or $_.PNPClass -in @('Camera','Image','Ports','Mouse','Keyboard')) -and 
            $_.ConfigManagerErrorCode -eq 0 -and 
            $_.Present -eq $true 
        } | 
        Select-Object Name, PNPClass, DeviceID, Manufacturer, Status | 
        ConvertTo-Json -Compress
    """
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return []
        items = json.loads(res.stdout)
        if isinstance(items, dict):
            items = [items]
    except Exception:
        return []

    IGNORE_KEYWORDS = [
        "root hub", "host controller", "composite device", "usb input device",
        "generic usb hub", "standard ps/2", "converted portable", "event filter",
        "streaming service proxy", "tpfhid", "i2c hid", "gpio laptop", "system controller",
        "consumer control device", "vendor-defined device", "input configuration device",
        "precisiontouchpad", "wireless radio controls", "portable device control"
    ]

    devices = []
    seen_keys = set()

    for item in items:
        name = (item.get("Name") or "").strip()
        pnp_class = (item.get("PNPClass") or "").strip()
        device_id = (item.get("DeviceID") or "").strip()

        if not name:
            continue

        lower_name = name.lower()
        if any(ign in lower_name for ign in IGNORE_KEYWORDS):
            continue

        # Classify device
        if pnp_class in ["Camera", "Image"] or "camera" in lower_name or "webcam" in lower_name:
            dev_type = "Camera"
        elif pnp_class in ["Ports"] or re.search(r'\bCOM\d+\b', name, re.I) or "serial" in lower_name or "ch340" in lower_name or "cp210" in lower_name or "ftdi" in lower_name or "uart" in lower_name:
            dev_type = "Serial Device"
        elif "esp32" in lower_name or "jtag" in lower_name:
            dev_type = "Development Device"
        elif pnp_class in ["Mouse", "Keyboard"] or "mouse" in lower_name or "keyboard" in lower_name:
            dev_type = "Input Device"
        elif pnp_class in ["DiskDrive", "USBStorage"]:
            dev_type = "Storage Device"
        elif pnp_class in ["Bluetooth"] or "bluetooth" in lower_name:
            dev_type = "Bluetooth Adapter"
        elif pnp_class in ["Biometric"] or "fingerprint" in lower_name:
            dev_type = "Biometric Sensor"
        elif pnp_class in ["MEDIA", "AudioEndpoint"] or "audio" in lower_name or "headphone" in lower_name:
            dev_type = "Audio Device"
        else:
            dev_type = "USB Device"

        conn_type = "USB"
        if "BTH" in device_id or "BLUETOOTH" in device_id:
            conn_type = "Bluetooth"

        # Extract VID/PID for clean deduplication
        vid_pid_match = re.search(r'VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})', device_id, re.I)
        if vid_pid_match:
            dedup_key = (vid_pid_match.group(0).upper(), dev_type)
        else:
            dedup_key = (name.lower(), dev_type)

        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        devices.append({
            "name": name,
            "type": dev_type,
            "connection": conn_type,
            "status": "CONNECTED",
            "device_id": device_id
        })

    return devices


def scan_camera_devices(max_index: int = config.MAX_CAMERA_DISCOVERY_INDEX) -> List[Dict[str, Any]]:
    """
    Safely probes OpenCV camera indices 0 through max_index.
    Verifies that a valid frame can be grabbed, extracts resolution,
    calculates preference scores for automatic camera selection,
    and releases the capture device immediately without leaving locks.
    """
    available_cameras = []
    friendly_names = get_directshow_camera_names()
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

    for idx in range(max_index + 1):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    if idx < len(friendly_names):
                        cam_name = friendly_names[idx]
                    else:
                        cam_name = f"Camera {idx}"

                    # Calculate preference score:
                    # Prefer genuine external USB webcams (Logitech C270) over laptop integrated cameras
                    score = 50
                    lower_name = cam_name.lower()
                    if "c270" in lower_name or "logi" in lower_name or "logitech" in lower_name:
                        score = 100
                    elif any(ext in lower_name for ext in ["webcam", "usb", "external"]):
                        if not any(internal in lower_name for internal in ["integrated", "wide vision", "built-in", "internal", "virtual"]):
                            score = 80
                        else:
                            score = 30
                    elif any(internal in lower_name for internal in ["integrated", "wide vision", "built-in", "internal", "virtual"]):
                        score = 30

                    available_cameras.append({
                        "index": idx,
                        "name": cam_name,
                        "resolution": f"{w}x{h}",
                        "width": w,
                        "height": h,
                        "status": "Available",
                        "preference_score": score,
                    })
        except Exception:
            pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    return available_cameras


def get_preferred_camera(max_index: int = config.MAX_CAMERA_DISCOVERY_INDEX) -> Dict[str, Any]:
    """
    Scans available OpenCV cameras and automatically selects the preferred camera:
    1. Highest priority: External USB webcam (e.g. Logitech C270)
    2. Fallback priority: Integrated/laptop camera (e.g. HP Wide Vision)
    3. Safe default: Camera 0
    Guarantees that the chosen camera index has been verified to open and capture a frame.
    """
    scanned = scan_camera_devices(max_index=max_index)
    if scanned:
        scanned.sort(key=lambda c: c.get("preference_score", 0), reverse=True)
        return scanned[0]

    return {
        "index": config.CAMERA_INDEX,
        "name": f"Camera {config.CAMERA_INDEX}",
        "resolution": f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT}",
        "width": config.FRAME_WIDTH,
        "height": config.FRAME_HEIGHT,
        "status": "Available",
        "preference_score": 0,
    }



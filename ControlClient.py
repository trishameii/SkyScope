from typing import Any, Dict, List, Union
import requests

class ClientRequests:

    def __init__(self, ip: str) -> None:

        self.ip = ip
        self.port = 8080

        self.server_addr = f"http://{self.ip}:{self.port}"

        r = requests.get(url=self.server_addr)
        
        assert r.content == b"Connected"

    def makeReqAndReturnJSON(self, route: str) -> Dict[str, Any]:
        try:
            r = requests.get(url=f"{self.server_addr}{route}")
            return r.json()
        except:
            pass

    # Camera Actions
    def startLiveStream(self, url) -> Dict[str, Any]:
        return self.makeReqAndReturnJSON(f'/startLiveStream/{url}')
    
    def stopLiveStream(self) -> Dict[str, Any]:
        return self.makeReqAndReturnJSON('/stopLiveStream')
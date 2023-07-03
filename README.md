nginx_handler.bat
-This batch file handles the closing NGINX.exe, it checks at a set time if 
 SkyScope.exe is still opened, if SkyScope is seen as closed, it will also 
 close NGINX.exe

launch_nginx_batch.vbs
-windows script that launches nginx_handler.bat as a hidden process so 
 that no window of it will pop up

launch_nginx.vbs
-windows script that launches NGINX.exe as a hidden process so that no
 window of it will pop up

ControlClient.py
-Python class that contains the requests module used to communicate and interface
 with the DJI Mini2 through the DJI SDK (RESTful API)
 
Folders for NGINX
 -conf
 -logs
 -site
 -temp

Folders and files for YOLOv5
 -data
 -utils
 -models
 -export.py

Folders and files for SkyScope
 -fonts
 -icons
 -img
 -SkyScope Projects
 -ControlClient.py
 -main.py
 -MainWindow.ui
 -SplashScreen.ui
 -best.pt
 -launch_nginx.vbs
 -launch_nginx_batch.vbs
 -nginx_handler.bat

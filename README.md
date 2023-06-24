In this project ML model predicts the fact that the user has performed a target event. (events like "Leave a request" and "Request a call"). Model is based on | GA_SESSIONS.csv - One line = one visit to the site | and | GA_HITS.csv - One line = one event per site visit |

All vizualisation in before_script.pynb | Also I made a little flask web-app, you can look at it by runnin these commands.
Linux/Mac os: export FLASK_APP=app.py  | flask run |||
Windows: set FLASK_APP=app.py | flask run  |||

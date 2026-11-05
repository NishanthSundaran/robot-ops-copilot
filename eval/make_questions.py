"""Writes eval/questions.jsonl: the evaluation set, written by hand from the corpus.

Each question has
  type     manual / manual_de / error / error_symptom / ros2 / moveit / driver / param / trials / unanswerable
  gold     where the answer is: [doc, page] pairs (PDF page, or section index for rst/md); used for retrieval recall
  facts    what a correct answer must contain: a list of groups, each a list of accepted spellings; every group
           must be present (numbers are compared with spaces removed and decimal commas read as points)
  split    dev (every third question; prompts and settings may be tuned on it) or test (reported)
German manual questions point at the English page; make_questions maps them to the German page with the key term
`de` (a phrase that occurs on that German page), and either page counts as a hit.

  python eval/make_questions.py
"""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from copilot.config import INDEX  # noqa: E402

EN = "UR3e_User_Manual_en_SW5.24.pdf"
DE = "UR3e_User_Manual_de_SW5.24.pdf"
ERR = "ErrorCodes.pdf"
R2 = "source/Tutorials/Beginner-CLI-Tools/"
R2C = "source/Tutorials/Beginner-Client-Libraries/"
MV = "doc/examples/"
UD = "ur_robot_driver/doc/"

Q = []


def add(type_, q, gold, facts, de=None, note=""):
    Q.append({"type": type_, "question": q, "gold": gold, "facts": facts, "de": de, "note": note})


# ---------------- UR3e user manual, English
add("manual", "How long is the cable between the Teach Pendant and the Control Box?", [[EN, 16]],
    [["4.5 m", "4.5m", "4.5 meter", "177 in"]])
add("manual", "What mains voltage range does the UR3e Control Box accept?", [[EN, 16]], [["100-240", "100 - 240", "100 to 240"]])
add("manual", "Which joints of the robot arm make the finer movements?", [[EN, 17]], [["wrist 1"], ["wrist 2"]])
add("manual", "After operation, how long should I let the robot cool down before touching it?", [[EN, 31]],
    [["one hour", "1 hour", "an hour", "60 min"]])
add("manual", "Which bolts and what tightening torque are used to mount the UR3e arm?", [[EN, 40]],
    [["m6"], ["9 nm", "9nm"]])
add("manual", "Which positioning pins do I use if the robot must be re-mounted accurately?", [[EN, 40]],
    [["iso 2338", "ø5 h6", "5 h6"]])
add("manual", "How do I enable Freedrive with the 3PE Teach Pendant?", [[EN, 60]],
    [["light-press", "light press"], ["release"], ["hold"]])
add("manual", "Why does the arm feel heavy to move in Freedrive near some positions?", [[EN, 60], [EN, 62]],
    [["resistance"], ["limit", "plane", "singularit"]])
add("manual", "In the Freedrive panel, what movement does the Plane option allow?", [[EN, 63]],
    [["x"], ["y"]])
add("manual", "After powering down and unplugging the robot, how long should I wait for stored energy to discharge?",
    [[EN, 63]], [["30 s", "30s", "30 seconds", "thirty seconds"]])
add("manual", "Which stop category does a Safeguard Stop perform?", [[EN, 84]], [["category 2", "stop category 2", "cat 2"]])
add("manual", "With external mode selection, is the robot in Automatic or Manual mode when the input is low?",
    [[EN, 84]], [["automatic"]])
add("manual", "What is the disagreement tolerance between the two input channels of a Three-Position Enabling Device?",
    [[EN, 90]], [["1 s", "1s", "1 second", "one second"]])
add("manual", "What supply does the remote ON/OFF control provide when the Control Box is off?", [[EN, 92]],
    [["12 v", "12v", "12-v"]])
add("manual", "What does the Tool Force safety limit restrict?", [[EN, 130]],
    [["force"], ["clamp"]])
add("manual", "What are the speed and force limits while the robot is in Recovery Mode?", [[EN, 200]],
    [["250 mm/s", "250mm/s", "0.25 m/s"], ["100 n", "100n"]])
add("manual", "What happens if a Recovery Mode safety limit is violated?", [[EN, 200]],
    [["category 0", "stop category 0", "cat 0"]])
add("manual", "During forced back-driving, how far at most should a joint be moved?", [[EN, 197]],
    [["160"]])
add("manual", "How do I move the robot into the Put in Box position for transport?", [[EN, 207]],
    [["installation"], ["home"], ["put in box"]])
add("manual", "How much free space is needed for the Control Box door, and how long can high voltage remain inside?",
    [[EN, 211]], [["915 mm", "915mm"], ["600 v", "600v", "several hours", "hours"]])
add("manual", "What does the Joint Load pane in the Log tab show for each joint?", [[EN, 216]],
    [["temperature"], ["load"], ["voltage"]])
add("manual", "How do I backdrive the arm using the 3PE Teach Pendant button?", [[EN, 201]],
    [["initialize", "on"], ["light-press", "light press"], ["backdrive"]])

# ---------------- UR3e user manual, German questions (answer in German, gold in either language)
add("manual_de", "Wie lang ist das Kabel zwischen Teach Pendant und Schaltschrank?", [[EN, 16]],
    [["4,5 m", "4.5 m", "4,5m", "4.5m"]], de="4,5 m")
add("manual_de", "Wie aktiviere ich Freedrive mit dem 3PE-Teach-Pendant?", [[EN, 60]],
    [["drück", "druck", "press"], ["halten", "hold", "gedrückt"]], de="Freedrive")
add("manual_de", "Mit welchen Schrauben und welchem Drehmoment wird der UR3e-Roboterarm montiert?", [[EN, 40]],
    [["m6"], ["9 nm", "9nm"]], de="9 Nm")
add("manual_de", "Wie lange soll der Roboter nach dem Betrieb abkühlen, bevor man ihn berührt?", [[EN, 31]],
    [["eine stunde", "1 stunde", "einer stunde", "60 min"]], de="Stunde")
add("manual_de", "Welche Geschwindigkeits- und Kraftgrenzen gelten im Wiederherstellungsmodus?", [[EN, 200]],
    [["250 mm/s", "250mm/s", "0.25 m/s"], ["100 n", "100n"]], de=[217])
add("manual_de", "Welche Stoppkategorie führt ein Schutzstopp (Safeguard Stop) aus?", [[EN, 84]],
    [["kategorie 2", "category 2", "stoppkategorie 2"]], de="Kategorie 2")
add("manual_de", "Wie viel Platz braucht die Tür des Schaltschranks zum vollständigen Öffnen?", [[EN, 211]],
    [["915 mm", "915mm"]], de="915 mm")
add("manual_de", "Wie lange soll man nach dem Ausstecken warten, bis gespeicherte Energie entladen ist?", [[EN, 63]],
    [["30 s", "30s", "30 sekunden"]], de="30 Sekunden")
add("manual_de", "Welche Toleranz haben die zwei Eingangskanäle eines Dreistellungs-Zustimmschalters?", [[EN, 90]],
    [["1 s", "1s", "1 sekunde", "eine sekunde"]], de="1 Sekunde")
add("manual_de", "Wie bringe ich den Roboter für den Transport in die Put-in-Box-Position?", [[EN, 207]],
    [["installation"], ["home", "startposition"]], de=[224, 225, 226])
add("manual_de", "Wie weit darf ein Gelenk beim erzwungenen Zurückfahren höchstens bewegt werden?", [[EN, 197]],
    [["160"]], de="160")
add("manual_de", "Welche Spannung liefert die Fern-EIN/AUS-Steuerung, wenn der Schaltschrank aus ist?", [[EN, 92]],
    [["12 v", "12v", "12-v"]], de="12V")
add("manual_de", "Was zeigt der Bereich Gelenklast im Protokoll-Tab für jedes Gelenk an?", [[EN, 216]],
    [["temperatur"], ["spannung"]], de="Temperatur")

# ---------------- UR error codes, code given
add("error", "The robot stopped with C158A1. What should I check?", [[ERR, "C158A1"]],
    [["payload"], ["center of gravity", "centre of gravity", "cog"]])
add("error", "What does C192A2 mean and what do I do about it?", [[ERR, "C192A2"]],
    [["emergency stop", "e-stop", "estop"], ["cable", "connection"]])
add("error", "We get C289A2 on the tool connector. What is going on?", [[ERR, "C289A2"]],
    [["sink current"], ["digital output"]])
add("error", "Explain error C261A5.", [[ERR, "C261A5"]], [["temperature"], ["low"]])
add("error", "PolyScope shows C351A5. How do I fix it?", [[ERR, "C351A5"]],
    [["cable extender"], ["unconfigure", "connect"]])
add("error", "What does C153A2 mean?", [[ERR, "C153A2"]],
    [["elbow"], ["path", "collision", "collison"]])
add("error", "How do I resolve C740A38?", [[ERR, "C740A38"]],
    [["48v", "48 v"], ["acceleration", "torque"]])
add("error", "What is C306A9 and how serious is it?", [[ERR, "C306A9"]],
    [["brake"], ["joint"]])
add("error", "C163A0 keeps appearing. What does UR recommend?", [[ERR, "C163A0"]],
    [["protective stop"], ["payload", "collision", "acceleration"]])
add("error", "What does C72A1 mean?", [[ERR, "C72A1"]], [["psu", "power supply"], ["48v", "48 v"]])
add("error", "What causes C204A3 and how do I avoid it?", [[ERR, "C204A3"]],
    [["stopj", "stopl"]])
add("error", "What happens after C348A1 and how do I recover?", [[ERR, "C348A1"]],
    [["external axes", "axis group"], ["reboot", "rebooting"]])
add("error", "What does the joint encoder warning C74A64 indicate?", [[ERR, "C74A64"]],
    [["magnetic ring"], ["far"]])
add("error", "Was bedeutet der Fehler C153A1 und was soll ich tun?", [[ERR, "C153A1"]],
    [["schulter", "shoulder"], ["nutzlast", "payload", "last", "schwerpunkt", "objekt", "hindernis"]])
add("error", "Fehlercode C72A1: Was ist zu prüfen?", [[ERR, "C72A1"]],
    [["netzteil", "psu", "stromversorgung", "power supply"], ["safety control board", "sicherheitssteuer"]])

# ---------------- UR error codes, symptom only (no code given: needs search)
add("error_symptom", "The robot reports that the 48V Safety Control Board supply voltage is below the minimum. "
    "What should I do?", [[ERR, "C740A38"]], [["acceleration", "torque"]])
add("error_symptom", "PolyScope says more than 50 protective stops were detected on the same joint. What does UR "
    "recommend?", [[ERR, "C163A0"]], [["collision"], ["payload"]])
add("error_symptom", "Our tool digital output keeps switching itself off. What does the error directory say?",
    [[ERR, "C289A2"]], [["sink current"]])
add("error_symptom", "What should I check for an emergency stop disagreement fault?", [[ERR, "C192A2"]],
    [["safety device", "cable", "connection"]])
add("error_symptom", "What can a joint encoder 'signal lost' warning mean?", [[ERR, "C74A32"]],
    [["readhead", "read head", "read-head"], ["ring"]])
add("error_symptom", "The controller says the robot cable extender is configured but not detected. What can I do?",
    [[ERR, "C351A5"]], [["unconfigure", "connect"]])
add("error_symptom", "The arm could not follow its path and the elbow joint reported it. What are the usual causes?",
    [[ERR, "C153A2"]], [["collision", "collison", "object"], ["payload", "acceleration", "center of gravity"]])

# ---------------- ROS 2 Humble docs
P = R2 + "Understanding-ROS2-Parameters/Understanding-ROS2-Parameters.rst"
add("ros2", "How do I start a node with parameter values loaded from a YAML file?", [[P, 9]],
    [["--params-file"], ["--ros-args"]])
add("ros2", "How do I save all current parameter values of a running node to a file?", [[P, 7]],
    [["ros2 param dump"]])
add("ros2", "Which command changes a node parameter at runtime?", [[P, 6]], [["ros2 param set"]])
C = R2C + "Colcon-Tutorial.rst"
add("ros2", "On a machine with little RAM, how do I make colcon build packages one at a time?", [[C, 8]],
    [["--executor sequential"]])
add("ros2", "How do I install colcon on Ubuntu?", [[C, 2]], [["python3-colcon-common-extensions"]])
add("ros2", "How do I run the tests of the packages in my colcon workspace?", [[C, 9]], [["colcon test"]])
add("ros2", "Which build system and build tool does ROS 2 package creation use?",
    [[R2C + "Creating-Your-First-ROS2-Package.rst", 1]], [["ament"], ["colcon"]])
Qs = "source/Concepts/Intermediate/About-Quality-of-Service-Settings.rst"
add("ros2", "Will a best effort publisher connect to a reliable subscription?", [[Qs, 4]],
    [["no", "not compatible", "incompatible", "won't", "will not"]])
add("ros2", "What reliability and queue size does the sensor data QoS profile use, and why?", [[Qs, 3]],
    [["best effort"], ["small"]])
add("ros2", "How can I keep ROS 2 traffic from being visible to other computers on the network?",
    [[R2 + "Configuring-ROS2-Environment.rst", 7]], [["ros_localhost_only"]])
add("ros2", "On Windows, which ROS domain IDs can be used without colliding with ephemeral ports?",
    [["source/Concepts/Intermediate/About-Domain-ID.rst", 3]], [["0-166", "0 - 166", "0 to 166"]])
B = R2 + "Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.rst"
add("ros2", "How do I record two topics into a bag file with a custom name?", [[B, 6]], [["ros2 bag record"], ["-o"]])
add("ros2", "How do I see the duration and message count of a recorded bag?", [[B, 7]], [["ros2 bag info"]])
L = R2 + "Using-Rqt-Console/Using-Rqt-Console.rst"
add("ros2", "How do I start a node so it only logs warnings and above?", [[L, 6]], [["--log-level"], ["warn"]])
add("ros2", "What are the ROS 2 logger levels, from most to least severe?", [[L, 5]],
    [["fatal"], ["error"], ["warn"], ["info"], ["debug"]])
add("ros2", "What tool should I use to publish a static transform instead of writing my own broadcaster?",
    [["source/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Static-Broadcaster-Py.rst", 9]],
    [["static_transform_publisher"]])
S = R2 + "Understanding-ROS2-Services/Understanding-ROS2-Services.rst"
add("ros2", "How do I list all active services together with their types?", [[S, 6]], [["ros2 service list -t",
                                                                                       "--show-types"]])
add("ros2", "In the output of ros2 interface show for a service type, what does the --- line separate?", [[S, 8]],
    [["request"], ["response"]])
add("ros2", "Wie setze ich die ROS_DOMAIN_ID für meine Knoten?", [[R2 + "Configuring-ROS2-Environment.rst", 6]],
    [["export ros_domain_id"]])

# ---------------- MoveIt 2
PZ = MV + "pilz_industrial_motion_planner/pilz_industrial_motion_planner.rst"
add("moveit", "Which motion types can the Pilz industrial motion planner generate?", [[PZ, 0]],
    [["ptp", "point-to-point", "point to point"], ["lin", "linear"], ["circ", "circular"]])
add("moveit", "What velocity profile does the Pilz PTP command use?", [[PZ, 4]], [["trapezoid"]])
add("moveit", "What does the Pilz planner need configured for LIN and CIRC motions?", [[PZ, 2]],
    [["cartesian"], ["limit"]])
MG = "doc/concepts/move_group.rst"
add("moveit", "Which topic does move_group listen to for the current joint state?", [[MG, 4]], [["/joint_states",
                                                                                                 "joint_states"]])
add("moveit", "Which interface does move_group use to send trajectories to the robot's controllers?", [[MG, 6]],
    [["followjointtrajectory"]])
add("moveit", "From which parameter does move_group read the SRDF?", [[MG, 2]], [["robot_description_semantic"]])
K = MV + "kinematics_configuration/kinematics_configuration_tutorial.rst"
add("moveit", "What is MoveIt's default kinematics plugin?", [[K, 4]], [["kdl"]])
add("moveit", "How do I enable position-only IK?", [[K, 7]], [["position_only_ik"]])
SV = MV + "realtime_servo/realtime_servo_tutorial.rst"
add("moveit", "Which message types does MoveIt Servo accept as input commands?", [[SV, 8]],
    [["twiststamped"], ["jointjog"]])
add("moveit", "What are the minimum requirements to run MoveIt Servo on my robot?", [[SV, 7]],
    [["urdf"], ["srdf"], ["encoder"]])

# ---------------- UR ROS 2 driver
ST = UD + "usage/startup.rst"
add("driver", "Which arguments are required when starting ur_control.launch.py?", [[ST, 1]],
    [["ur_type"], ["robot_ip"]])
add("driver", "Which URCap do I need on an e-Series robot for the ROS 2 driver, and what PolyScope version?",
    [[UD + "installation/install_urcap_e_series.rst", 0]], [["externalcontrol", "external control"], ["5.1"]])
RI = UD + "ROS_INTERFACE.md"
add("driver", "What does headless_mode do in the UR driver?", [[RI, 3]],
    [["urscript", "script"], ["remote"]])
add("driver", "What is the allowed range of servoj_gain?", [[RI, 6]], [["100"], ["3000"]])
add("driver", "What happens if the robot's calibration does not match the kinematics hash?", [[RI, 4]],
    [["error"]])
add("driver", "How do I extract the factory calibration from my UR robot?", [["ur_calibration/README.md", 1]],
    [["calibration_correction"]])
add("driver", "What is the simplest way to get a lower-latency kernel for the UR driver?",
    [[UD + "installation/real_time.rst", 1]], [["linux-lowlatency"]])
add("driver", "What does the speed_scaling_state_broadcaster publish?", [["ur_controllers/doc/index.rst", 3],
                                                                         ["ur_controllers/doc/index.rst", 0]],
    [["speed"], ["0"], ["1"]])
add("driver", "The driver prints 'Connection to reverse interface dropped'. What does that mean?", [[ST, 4]],
    [["external control"], ["restart", "unpause", "interrupted"]])
add("driver", "Can I use the rs485 tool communication URCap together with the Robotiq Grippers URCap?",
    [[UD + "setup_tool_communication.rst", 0]], [["issue", "not", "refrain"]])

# ---------------- thesis parameters (YAML of the UR3e stack)
SM = "safety_monitor.yaml"
add("param", "What is the maximum hand guiding force in the safety monitor?", [[SM, 1], [SM, 0]], [["150"]])
add("param", "At what distance does the safety monitor command a full stop?", [[SM, 0]], [["0.12"]])
add("param", "What quasi-static force limit does the safety monitor use for power and force limiting?",
    [[SM, 0], [SM, 1]], [["140"]])
add("param", "What robot stopping time does the SSM calculation assume?", [[SM, 0], [SM, 1]], [["0.4"]])
add("param", "Which human approach speed is assumed for speed and separation monitoring?", [[SM, 0], [SM, 1]],
    [["1.6"]])
add("param", "What is the absolute minimum separation distance before a protective stop?", [[SM, 0], [SM, 1]],
    [["0.1", "0.10"]])
add("param", "At what rate does the safety monitor run?", [[SM, 0]], [["100"]])
add("param", "What update rate is the controller manager configured with?", [["moveit_controllers.yaml", 0]],
    [["500"]])
add("param", "Welche maximale Handführungsgeschwindigkeit ist im Safety Monitor eingestellt?", [[SM, 1], [SM, 0]],
    [["0.25", "0,25"]])

# ---------------- thesis trial logs (needs the SQL tool; no retrieval gold)
add("trials", "How many trials were logged in total?", [], [["38"]])
add("trials", "How many trials ended with a successful grasp?", [], [["28"]])
add("trials", "What was the grasp success rate over all logged trials?", [], [["73.7", "73.68", "74", "73,7", "28/38",
                                                                                    "28 of 38", "28 out of 38"]])
add("trials", "What was the average total time of the successful trials?", [], [["41.3"]])
add("trials", "Which trial had the highest peak force, and how high was it?", [], [["20260422_172522"], ["47.06",
                                                                                                         "47.1"]])
add("trials", "What was the mean peak force over the successful grasp trials?", [], [["7.26", "7.3"]])
add("trials", "Which trial took the longest, and how long?", [], [["20260320_154132"], ["64.3"]])
add("trials", "How many trials were run on 21 April 2026?", [], [["13"]])
add("trials", "On which day were the most trials run?", [], [["2026-04-21", "20260421", "21 april", "april 21",
                                                               "21.04"]])
add("trials", "What is the most frequent state transition in the logs?", [], [["idle"], ["nudge"]])
add("trials", "Which state does the controller go to after GUIDING?", [], [["idle"]])
add("trials", "How many trials had a peak force above 20 N?", [], [["4"]])
add("trials", "On average, how long did trials spend in the AWAIT_INTERACTION phase?", [], [["9.56", "9.6"]])
add("trials", "Wie viele Versuche endeten mit einem erfolgreichen Griff?", [], [["28"]])

# ---------------- not answerable from the corpus: the copilot must say so
for q in ["How do I connect the UR3e to a KUKA KR C4 controller?",
          "What is the maximum payload of the UR20?",
          "What does a UR3e cost?",
          "How do I update the firmware of a Robotiq 2F-85 gripper?",
          "What does error code C9999A1 mean?",
          "What was the peak force in trial 20250101_000000?",
          "How do I configure MoveIt for a Fanuc LR Mate?",
          "Wie kalibriere ich einen ABB IRB 120?",
          "What torque should the base bolts of a UR10e be tightened to?",
          "What is the value of the max_joint_jerk parameter in the thesis stack?",
          "Which Python version does PolyScope X use?",
          "What is the IP rating of the KUKA LBR iiwa?",
          "How many trials did the thesis run with the Franka Panda?",
          "Was kostet ein Wartungsvertrag für den UR3e?"]:
    add("unanswerable", q, [], [])


# Accepted spellings added after reviewing cycle-1 answers by hand. Each was a correct answer the grader missed;
# nothing was added for a wrong answer:  0.25 m/s (= 250 mm/s), "read head" (= readhead), 73.68 (= 73.7 %).
GRADER_CHANGES = ["0.25 m/s for 250 mm/s (q015, q026)", "read head for readhead (q054)", "73.68 for 73.7 (q107)",
                  "script for URScript (q088, cycle 2: 'sends the script code directly')",
                  "12-v for 12 V (q033, cycle 3: '12-V-Hilfsstromversorgung')"]


def en_to_de_pages(chunks):
    """English manual questions also accept the German page with the same content: a page in the same numbered
    section whose text contains the answer's number (decimal comma). Retrieving it is a correct retrieval."""
    import re
    for q in Q:
        if q["type"] != "manual":
            continue
        en_page = q["gold"][0][1]
        sec = next((c["title"] for c in chunks if c["doc"] == EN and c["page"] == en_page), "")
        num = re.match(r"^\d+(\.\d+)*", sec)
        keys = [x.replace(".", ",") for g in q["facts"] for x in g if re.search(r"\d", x)]
        if not keys or not num:
            continue
        pages = sorted({c["page"] for c in chunks if c["doc"] == DE and c["title"].startswith(num.group(0) + ".")
                        and any(k in c["text"] for k in keys)})
        if 0 < len(pages) <= 3:
            q["gold"] += [[DE, p] for p in pages]


def error_pages():
    """Error questions name a code; the index stores one chunk per code group, so gold = that group's page."""
    e = json.loads((INDEX / "errors.json").read_text())
    gp = {g["code"]: g["page"] for g in e["groups"]}
    grp = {c["code"]: c["group"] for c in e["codes"]}
    for q in Q:
        q["gold"] = [[d, gp[grp[p]]] if d == ERR and isinstance(p, str) else [d, p] for d, p in q["gold"]]


def de_pages(chunks):
    """German question -> also accept the German page that contains its key term, in the same section."""
    import re
    for q in Q:
        if q["type"] != "manual_de":
            continue
        if isinstance(q["de"], list):
            q["gold"] += [[DE, p] for p in q["de"]]
            q["note"] = f"de pages {q['de']} (set by hand)"
            continue
        en_page = q["gold"][0][1]
        sec = next((c["title"] for c in chunks if c["doc"] == EN and c["page"] == en_page), "")
        num = re.match(r"^\d+(\.\d+)*", sec)
        cands = [c for c in chunks if c["doc"] == DE and q["de"] in c["text"]
                 and (not num or c["title"].startswith(num.group(0) + "."))]
        if not cands:
            cands = [c for c in chunks if c["doc"] == DE and q["de"] in c["text"]]
        pages = sorted({c["page"] for c in cands})
        q["gold"] += [[DE, p] for p in pages[:3]]
        q["note"] = f"de pages {pages[:3]} via '{q['de']}', section {sec}"


def main() -> None:
    chunks = [json.loads(ln) for ln in open(INDEX / "chunks.jsonl")]
    de_pages(chunks)
    en_to_de_pages(chunks)
    error_pages()
    out = REPO / "eval" / "questions.jsonl"
    with open(out, "w") as f:
        for i, q in enumerate(Q):
            q = {"id": f"q{i:03d}", "split": "dev" if i % 3 == 0 else "test", **q}
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    by = {}
    for q in Q:
        by[q["type"]] = by.get(q["type"], 0) + 1
    print(len(Q), by)
    for q in Q:
        if q["type"] == "manual_de":
            print(q["note"])
    # every gold (doc, page) must exist in the index
    have = {(c["doc"], c["page"]) for c in chunks}
    missing = [(q["question"][:50], g) for q in Q for g in q["gold"] if tuple(g) not in have]
    print("gold not in index:", missing or "none")


if __name__ == "__main__":
    main()

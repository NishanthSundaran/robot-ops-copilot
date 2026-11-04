"""Writes eval/questions_heldout.jsonl: 50 new questions, written after the first evaluation round but before any
fix was made or any output of the fixed system was seen, from pages and docs the first set does not use.
The fixes of cycle 2+ were driven by failures on eval/questions.jsonl; this set measures whether they generalise.

Gold pages are resolved, not typed: ("find", doc substring, phrase) = every chunk page of that document whose text
contains the phrase.

  python eval/make_heldout.py
"""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from copilot.config import INDEX  # noqa: E402

EN = "UR3e_User_Manual_en_SW5.24.pdf"
DE = "UR3e_User_Manual_de_SW5.24.pdf"
Q = []


def add(type_, q, gold, facts):
    Q.append({"type": type_, "question": q, "gold": gold, "facts": facts})


def f(doc, phrase):
    return ("find", doc, phrase)


# UR3e manual, English (pages not used by the first set)
add("manual", "What are the maximum payload and the reach of the UR3e?", [f(EN, "Maximum payload 3 kg")],
    [["3 kg", "3kg"], ["500 mm", "500mm"]])
add("manual", "What is the IP classification of the UR3e?", [f(EN, "IP54")], [["ip54"]])
add("manual", "In what ambient temperature range can the UR3e operate, and what happens above 35 °C?",
    [f(EN, "Ambient temperature range")], [["0-50", "0 - 50", "0 to 50"], ["reduced"]])
add("manual", "How accurate is the UR3e force torque sensor?", [f(EN, "Force Torque sensor accuracy")],
    [["3.5 n", "3.5n"]])
add("manual", "Why should the tool not be moved close to the cylinder directly above or below the robot base?",
    [f(EN, "cylindrical volume")], [["fast"]])
add("manual", "How do I make sure the robot cable connector is properly locked?", [f(EN, "Twist the connector twice")],
    [["twist"], ["twice", "two times"]])
add("manual", "What kind of converter do I need to connect an HDMI monitor to the Control Box Mini DisplayPort?",
    [f(EN, "Passive converters do not work")], [["active"]])
add("manual", "Which supply voltages can the tool I/O power supply be set to?", [f(EN, "internal power supply to 0V")],
    [["12"], ["24"]])
add("manual", "What changes once an operational mode password is set?", [f(EN, "Operational Mode Password")],
    [["manual"], ["automatic"]])
add("manual", "Can I resume a program after pressing Stop?", [f(EN, "You cannot resume a stopped program")],
    [["cannot", "can't", "can not", "not possible", "no"], ["play"]])
add("manual", "What does the MODBUS sequential mode option do?", [f(EN, "Sequential mode")],
    [["wait"], ["response"]])
add("manual", "Which cleaning agents may be used on the robot arm?", [f(EN, "70% Isopropyl")],
    [["isopropyl"]])
add("manual", "How long can exporting the technical support file take?", [f(EN, "up to 10 minutes")],
    [["10 min", "10min", "ten min"]])
add("manual", "Where are the pinch hazard zones around the robot base?", [f(EN, "450 mm from the base")],
    [["450"], ["200"]])

# German questions (gold in the German manual)
add("manual_de", "Wie hoch ist die maximale Nutzlast des UR3e?", [f(DE, "Maximale Nutzlast 3 kg"), f(EN, "Maximum payload 3 kg")],
    [["3 kg", "3kg"]])
add("manual_de", "Welche IP-Schutzart hat der UR3e?", [f(DE, "IP54"), f(EN, "IP54")], [["ip54"]])
add("manual_de", "Mit welchen Reinigungsmitteln darf der Roboterarm gereinigt werden?",
    [f(DE, "Isopropyl"), f(EN, "70% Isopropyl")], [["isopropyl"]])
add("manual_de", "Wie lange kann der Export der Support-Datei dauern?", [f(DE, "10 Minuten"), f(EN, "up to 10 minutes")],
    [["10 min"]])
add("manual_de", "Welche Spannungen kann die Werkzeug-Stromversorgung liefern?",
    [f(DE, "0 V, 12 V oder 24 V"), f(DE, "12 V/24 V"), f(EN, "internal power supply to 0V")], [["12"], ["24"]])

# error codes
add("error", "What does C192A15 mean?", [f("ErrorCodes.pdf", "C192A15")], [["reduced mode"], ["disagreement"]])
add("error", "How do I fix C264A10?", [f("ErrorCodes.pdf", "C264A10")], [["reboot"]])
add("error", "What is error C353A6?", [f("ErrorCodes.pdf", "C353A6")], [["imu"], ["timeout"]])
add("error", "We see C159A5 on the robot. What is it?", [f("ErrorCodes.pdf", "C159A5")],
    [["wrist 3"], ["payload"]])
add("error", "What should I check for C283A153?", [f("ErrorCodes.pdf", "C283A153")],
    [["input"], ["connect"]])
add("error", "Was bedeutet Fehler C305A0?", [f("ErrorCodes.pdf", "C305A0")],
    [["power control", "stromsteuerung", "leistungssteuerung", "energiesteuer", "power-control"]])
add("error_symptom", "PolyScope warns that the user specified payload is 0 kg on Wrist 3. What should I do?",
    [f("ErrorCodes.pdf", "C159A5")], [["payload"], ["center of gravity", "centre of gravity"]])
add("error_symptom", "A Keep-Alive message with a wrong sequence was received from the Safety Control Board. What "
    "does UR suggest?", [f("ErrorCodes.pdf", "C309A1")], [["reboot"]])
add("error_symptom", "Too many misaligned Start of Cycle pulses were detected. What could cause this?",
    [f("ErrorCodes.pdf", "C264A10")], [["damaged", "reboot"]])

# ROS 2 docs
TOP = "Understanding-ROS2-Topics"
add("ros2", "How do I see the rate at which a topic is published?", [f(TOP, "ros2 topic hz")], [["ros2 topic hz"]])
add("ros2", "How do I list the actions in my system together with their types?",
    [f("Understanding-ROS2-Actions", "ros2 action list -t")], [["ros2 action list -t"]])
add("ros2", "How do I see a node's subscribers, publishers, services and actions?",
    [f("Understanding-ROS2-Nodes", "ros2 node info")], [["ros2 node info"]])
add("ros2", "How do I rename a node when starting it from the command line?",
    [f("Understanding-ROS2-Nodes", "__node:=")], [["__node:="]])
add("ros2", "How do I print the messages published on a topic?", [f(TOP, "ros2 topic echo")], [["ros2 topic echo"]])

# MoveIt 2
add("moveit", "Which planner library does MoveIt use by default?", [f("motion_planning.rst", "OMPL")], [["ompl"]])
add("moveit", "What does CHOMP stand for?", [f("chomp_planner", "Covariant Hamiltonian")],
    [["covariant hamiltonian optimization"]])
add("moveit", "Which tool generates the MoveIt configuration package for a new robot?",
    [f("setup_assistant", "Setup Assistant")], [["setup assistant"]])

# UR driver
add("driver", "Which launch argument runs the UR driver without a real robot?",
    [f("ur_robot_driver", "use_fake_hardware"), f("ur_robot_driver", "use_mock_hardware")], [["use_fake_hardware", "use_mock_hardware"]])
add("driver", "What does the controller_stopper node do?", [f("controller_stopper", "controller")],
    [["stop"], ["controller"]])
add("driver", "How can I run URSim for the driver without installing it?", [f("ursim_docker", "docker")],
    [["docker"]])

# thesis parameters and trials
add("param", "What system reaction time does the safety monitor's SSM calculation use?",
    [f("safety_monitor.yaml", "T_reaction")], [["0.15"]])
add("param", "What depth uncertainty of the RealSense is assumed in the safety monitor?",
    [f("safety_monitor.yaml", "Z_sensor")], [["0.04"]])
add("trials", "How many trials ended without a successful grasp?", [], [["10"]])
add("trials", "What was the mean object displacement over all trials?", [], [["0.11"]])
add("trials", "How many trials were logged on 8 April 2026?", [], [["9"]])

# not answerable
for q in ["What is the maximum payload of the UR5e?", "What is the mean time between failures of the UR3e gearboxes?",
          "Wie programmiere ich eine Siemens S7 mit dem TIA Portal?",
          "Which error code does the UR3e show when the vacuum gripper loses air pressure?",
          "In which trial was the Robotiq gripper set to 50 % force?"]:
    add("unanswerable", q, [], [])


def main() -> None:
    chunks = [json.loads(ln) for ln in open(INDEX / "chunks.jsonl")]
    bad = []
    for q in Q:
        gold = []
        for g in q["gold"]:
            _, doc, phrase = g
            if doc == "ErrorCodes.pdf":  # error codes are indexed per group: find the group chunk
                pages = sorted({c["page"] for c in chunks if c["source"] == "ur_errors" and phrase in c["text"]})
                gold += [["ErrorCodes.pdf", p] for p in pages]
                continue
            pages = sorted({(c["doc"], c["page"]) for c in chunks if doc in c["doc"] and phrase in c["text"]})
            gold += [list(p) for p in pages]
        if q["gold"] and not gold:
            bad.append(q["question"])
        q["gold"] = gold
    with open(REPO / "eval" / "questions_heldout.jsonl", "w") as fh:
        for i, q in enumerate(Q):
            fh.write(json.dumps({"id": f"h{i:03d}", "split": "heldout", **q}, ensure_ascii=False) + "\n")
    by = {}
    for q in Q:
        by[q["type"]] = by.get(q["type"], 0) + 1
    print(len(Q), by)
    print("unresolved gold:", bad or "none")


if __name__ == "__main__":
    main()

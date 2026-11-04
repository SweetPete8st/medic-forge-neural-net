# For paramedic instructors: what this project is trying to build, in plain language

**The problem.** Writing a good simulation scenario takes hours: the patient's story, the starting
vitals, how the patient gets worse if students do nothing, how each drug and device changes things,
what the patient says, and what counts as a good run. Then it has to behave the same way every
time so that two students can be compared fairly.

**The idea being tested.** Instead of writing all of that by hand, an instructor writes a short
description ("34-year-old with a bad asthma attack after cleaning with bleach, moderate to severe on
arrival, I want students to recognise fatigue and avoid sedation"). A computer program running on an
academy machine turns that into a **patient sheet**: a readable document that lists every number and
every rule the patient will follow, each with where the number came from. The instructor and a
clinical reviewer read and correct that sheet. The simulator then runs the sheet.

**What the patient sheet looks like.** One file per scenario, in plain text, with sections for the
patient's baseline, the hidden physiology (how blocked the airways are, how tired the breathing
muscles are, blood gases), each drug's onset, peak and duration, each device (oxygen, CPAP,
bag-valve-mask), the stages of deterioration, the expected path for "untreated", "treated promptly",
"treated late" and "treated wrongly", what the monitor shows and when, what the patient can say,
and a sign-off block. Nothing in the simulator is hidden from the reviewer: if a number is wrong, you
change it in the sheet.

**What stays the same across scenarios.** The physiology engine. It is one piece of software,
checked once, that knows how obstruction, breathing effort, carbon dioxide, oxygen, heart rate, blood
pressure and consciousness move together. Each scenario only changes the numbers and a few rules.
That is why a new scenario can be cheap: you are not building a new patient from scratch, you are
filling in a sheet for a patient the engine already knows how to run.

**What we found so far.**
- The engine runs a coherent asthma patient. Untreated, the patient gets worse the way the guidelines
  describe (can only say words, carbon dioxide rises, chest goes quiet, consciousness drops, then
  arrest at about 30 minutes). Treated early, they recover. Sedated by mistake, or bagged too fast,
  they get worse in the way you would expect.
- Every pair of interventions at every stage was run automatically (287 runs) and nothing impossible
  happened (no negative pressures, no oxygen that lowers oxygen). One real bug was found this way and
  fixed in one line.
- The same patient run twice from the same student actions gives the identical log. That is the
  repeatability you need for fair assessment.
- The idea of having an AI write the patient as a neural network instead of as readable rules was
  tested on a simple drug-dosing model. It worked after one correction, but it was no more accurate,
  no faster, and much harder to check than the readable version. For now the recommendation is the
  readable version.

**A second step, tried the same day.** We asked whether the whole patient could be turned into a
neural network automatically, not by an AI guessing numbers but by a compiler that reads the readable
rules and produces the network, the way a compiler turns source code into a program. It works: the
compiled "patient brain" follows the same path as the readable engine to within about half a minute
over an hour-long session, and the same rules always produce exactly the same network. What it does
not yet do is anything the readable engine cannot. It is a different packaging of the same patient,
useful if we ever need every patient to be one small file that any computer can run, or if we want to
nudge a patient's parameters automatically from an instructor's corrections (being tested).

**What is not done.**
- The asthma sheet's drug numbers (how fast albuterol works, how long it lasts, how much it raises
  the heart rate) are the author's working knowledge, not yet checked against the drug labels and
  GINA. They are marked "unverified" in the sheet so a reviewer can go through them in order.
- No instructor has used it yet. The next step is to time real instructors writing a scenario the
  old way and the new way, including review time.
- The talking patient (a small language model that speaks only from the sheet) is designed but not
  wired in.

**What we need from instructors and clinicians.** Half a day from a paramedic educator and a
physician to review the asthma sheet, line by line, and sign the review block. Then two instructors
willing to build a scenario each with the new method while we time them.

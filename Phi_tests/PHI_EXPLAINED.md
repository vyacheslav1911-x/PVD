# Φ Explained From Zero

Every term, from the raw numbers up. No step skipped. Every number in this
document was measured on the real SO-101 arm or computed from the real logs —
nothing is illustrative or made up.

---

## Table of contents

0. [The raw material: what data we even have](#0)
1. [Units: degrees, radians, ticks, percent](#1)
2. [Chunks: what a "chunk" is and why 50](#2)
3. [d1 and d2: the two most important quantities](#3)
4. [R — the dither ratio](#4)
5. [Reversal rate](#5)
6. [Frequency from scratch: what the FFT does](#6)
7. [The window function](#7)
8. [fracE_bw — energy the motor cannot follow](#8)
9. [p99_v and p99_a — how fast, how hard](#9)
10. [min_invk — how close to a singularity](#10)
11. [seam_v — the join between chunks](#11)
12. [The gates: S_pos, S_vel, S_env](#12)
13. [RNEA — where torque numbers come from](#13)
14. [D_demo and Mahalanobis distance](#14)
15. [The final Φ, assembled](#15)
16. [Things we tried and threw away](#16)
17. [Known problems — read before trusting any of this](#17)

---

<a name="0"></a>
## 0. The raw material: what data we even have

Everything starts as a **table of numbers**. One row per control tick, one
column per motor.

The arm has 6 motors. In LeRobot's order:

```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

The robot runs at about 30 ticks per second, so one row = 1/30 s = 33.4 ms.

A recorded rollout file (`record_run1.csv`) has FOUR blocks of six columns:

| prefix | meaning |
|---|---|
| `cmd_` | what the policy **asked for** |
| `sent_` | what actually **reached the motors** (cmd_ after a safety clamp) |
| `pos_` | what the arm **actually did** (Present_Position) |
| `velraw_` | the motor's own velocity register |

**This distinction is the single most important thing in the whole document.**
`cmd_` and `pos_` are very different. The policy can ask for something the arm
cannot do; the arm just fails to do it. Measured on our data: the largest gap
between what was asked and what was achieved was **50 degrees**.

Φ scores `cmd_`, because the whole point is to judge a trajectory *before*
running it.

A note on `velraw_`: it looks like it would save us from having to compute
velocity. It doesn't. It is an integer register with only ~50 distinct values
across an entire run and it reads exactly zero 18–47% of the time. Unusable.
Velocity has to be computed from position differences.

---

<a name="1"></a>
## 1. Units: degrees, radians, ticks, percent

Four different unit systems appear, and mixing them silently is how you get
wrong answers.

**Raw ticks.** The motor has a 12-bit magnetic encoder: 4096 steps per full
revolution. So one tick = 360/4096 = **0.088 degrees**. This is the finest
distinction the hardware can make. Anything smaller is invisible.

**Degrees.** LeRobot converts ticks to degrees using the calibration file:

```
degrees = (raw_ticks − mid) × 360 / 4095
mid = (range_min + range_max) / 2
```

`range_min` and `range_max` come from calibrating the arm — they are where the
joint physically stops. Note `mid` is defined as **0 degrees**, so degrees are
measured from the middle of each joint's travel.

Real values from this arm's calibration file:

| joint | range_min | range_max | mid | span |
|---|---|---|---|---|
| shoulder_pan | 831 | 3353 | 2092 | 221.7° |
| shoulder_lift | 807 | 3208 | 2007 | 211.0° |
| elbow_flex | 849 | 3095 | 1972 | 197.4° |
| wrist_flex | 776 | 3262 | 2019 | 218.5° |
| wrist_roll | 0 | 4095 | 2047 | 359.9° |
| gripper | 1342 | 2854 | 2098 | 132.9° |

**Percent.** The gripper is NOT in degrees. It is 0–100, "percent open".

**Radians.** The physics engine (Pinocchio) wants radians. So:
- the 5 body joints: `radians = degrees × π/180`
- the gripper: `radians = percent × 0.0191986`
  (from common.py: 0% maps to −0.174533 rad, 100% maps to 1.74533 rad, so
  each percent is (1.74533 + 0.174533)/100 = 0.0191986 rad)

In the code this is one array:

```python
SCALE = [π/180, π/180, π/180, π/180, π/180, 0.0191986]
```

Multiply the whole table by it and everything is in radians.

**Why this matters:** if you forget the gripper is percent and treat it as
degrees, you scale it wrong by a factor of ~1.1, and it silently poisons every
joint-averaged number.

---

<a name="2"></a>
## 2. Chunks: what a "chunk" is and why 50

The policy does not output one action at a time. It outputs a **chunk** — a
block of 50 future actions, predicted all at once. The robot executes them one
per tick, then the policy produces a new chunk.

In the log there's a column `chunk_start` that is 1 on the tick a new chunk
begins. Reading it naively gives chunk lengths of `1, 49, 1, 49, 1, 49...`

That pattern is a quirk of the recorder: it flags **both** the boundary tick and
the one after it. So the real structure is: a chunk starts, runs 50 ticks, next
chunk starts. To recover real boundaries, collapse consecutive flags:

```python
idx = where(chunk_start == 1)
starts = [i for k, i in enumerate(idx) if k == 0 or i - idx[k-1] > 1]
```

50 ticks at 30 Hz = **1.67 seconds** of planned motion per chunk.

Everything in Φ is computed **per chunk**, because that is the unit the policy
actually produces and the unit you would accept or reject.

For the demonstration dataset, which is continuous teleoperation with no chunks,
we cut it into 50-frame windows so the numbers are comparable.

---

<a name="3"></a>
## 3. d1 and d2: the two most important quantities

Almost every term is built from these two.

**d1 = the difference between consecutive actions.**

```
d1[i] = a[i+1] − a[i]
```

From 50 actions you get 49 values of d1. d1 is **how far the joint moves in one
tick**. It is velocity, just not divided by time yet.

**d2 = the difference between consecutive d1's.**

```
d2[i] = d1[i+1] − d1[i]
```

From 49 d1's you get 48 d2's. d2 is **how much the movement changed**. It is
acceleration, again not divided by time.

To get real physical units:
```
velocity     = d1 / dt        dt = 1/30 s = 0.0334 s
acceleration = d2 / dt²
```

**Worked example.** A joint at these angles over 5 ticks:

```
angles:  10.0   10.5   11.0   11.5   12.0
d1:         0.5   0.5   0.5   0.5           moves 0.5 deg every tick
d2:            0.0   0.0   0.0              never changes its rate
```
Smooth, steady motion. d2 is all zero.

Now a jittery joint:

```
angles:  10.0   11.0   10.2   11.1   10.3
d1:         1.0  -0.8   0.9  -0.8            forward, back, forward, back
d2:           -1.8   1.7  -1.7               huge changes every tick
```

Notice: **d1 is about the same size in both cases (0.5 vs ~0.9)** — both joints
are "moving" a similar amount. But d2 is 0 in the first and ~1.7 in the second.
d2 is what separates smooth motion from thrashing.

---

<a name="4"></a>
## 4. R — the dither ratio

### The formula

```
R = mean(|d2|) / mean(|d1|)
```
computed **for each joint separately**, then averaged over the six joints.

### What it means

The top is "how much it changes its mind." The bottom is "how far it actually
gets." R is the ratio of the two.

Using the two examples from section 3:
- smooth joint: mean|d2| = 0, mean|d1| = 0.5 → **R = 0**
- jittery joint: mean|d2| = 1.73, mean|d1| = 0.875 → **R = 1.98**

### Why compute it per joint and then average

R is **scale-invariant per joint**: if you doubled all of one joint's numbers,
both top and bottom double and R is unchanged. That means it doesn't matter
whether you feed it degrees, radians, or ticks — you get the same R.

But that only holds *within* one joint. If you pooled all six joints together
before dividing, the gripper's percent-units would contaminate the degrees. So:
divide first, average after.

### The deeper meaning: R is a ratio of times

Substitute d1 = v·dt and d2 = a·dt²:

```
R = (mean|a| · dt²) / (mean|v| · dt) = dt · mean|a| / mean|v|
```

Now `mean|v| / mean|a|` has units of **seconds**. Call it τ_change — how long
the motion takes to substantially change what it is doing. Then:

```
R = dt / τ_change
```

**R is your control period divided by the trajectory's natural timescale.**

- R < 1: the motion persists longer than one tick. Momentum carries it. Normal.
- R = 1: the velocity completely turns over inside one tick.
- R > 1: it turns over more than completely — it reverses.

### The measured numbers

dt = 33.4 ms, so τ_change = 33.4/R:

| | R | τ_change |
|---|---|---|
| human demonstrations | **0.619** | 54 ms |
| pi0.5 | **0.714** | 47 ms |
| SmolVLA | **1.540** | **22 ms** |

Humans revise their motion every ~54 ms, about 1.6 ticks. SmolVLA revises every
22 ms — **less than one control tick**. It is issuing commands that contradict
each other faster than the controller can even send them.

For context, the servo's own response time (measured, section 12) is about
125 ms. SmolVLA changes its mind roughly 5 times before the motor has finished
reacting to the first command.

### Separation

SmolVLA 1.29–1.75, pi0.5 0.33–1.13. **Completely disjoint. AUC = 1.000,
p = 1.6e-08.** The single best discriminator we found.

### The critical limitation

**R is a mean, so it dilutes single events.** We tested this by injecting one
violent reversal into an otherwise clean human chunk:

| injected spike | R |
|---|---|
| 0° | 0.245 |
| 4° | 0.283 |
| 25° | 0.393 |

A 25-degree snap barely moves R, because it's one event averaged over 50 frames.
**R catches sustained chattering. It does not catch isolated violence.**
That is what p99_a (section 9) is for.

### Where R does NOT generalize

R has no reference to the robot. "R > 1 is bad" is true for a slow arm doing a
slow reach. A balancing humanoid legitimately reverses fast and would score
R > 1 while behaving correctly. **Never port the threshold; always compare R to
demonstrations of the same task.**

---

<a name="5"></a>
## 5. Reversal rate

### The formula

```
reversal_rate = fraction of ticks where d1 changes sign
              = mean( sign(d1[i]) × sign(d1[i+1]) < 0 )
```

If the joint was going up and now goes down, that's a reversal. Count them,
divide by the number of opportunities.

### Why it exists

It is the most literally interpretable version of "the joint keeps changing
direction". No ratio, no FFT, no units.

### The measured numbers

| | shoulder_pan | shoulder_lift | elbow | wrist_flex | wrist_roll |
|---|---|---|---|---|---|
| demos | 0.005 | 0.005 | 0.016 | 0.019 | 0.007 |
| pi0.5 | 0.286 | 0.358 | 0.215 | 0.223 | 0.331 |
| SmolVLA | 0.556 | 0.551 | 0.564 | 0.554 | 0.587 |

**A human changes a joint's direction on 1% of ticks. SmolVLA does it on 56% —
every other tick, on every joint.**

AUC 1.000 SmolVLA vs pi0.5; AUC 0.000 demos vs SmolVLA (perfectly separated in
both directions).

Same generalization warning as R: a balancing controller would legitimately
have a high reversal rate.

---

<a name="6"></a>
## 6. Frequency from scratch: what the FFT does

This section assumes you know nothing about Fourier transforms.

### What "frequency" means here

Take a joint's angle over 50 ticks. It wiggles. Some of that wiggling is slow
(the arm sweeping across to grab something). Some is fast (buzzing in place).
"Frequency" is just **how fast a particular wiggle is**, in wiggles per second
(Hz).

The FFT sorts the total wiggling into slow and fast parts. Think of the bars on
a music equalizer: bass on the left, treble on the right. Same thing.

### Which frequencies get tested, and why exactly those

You do not get to pick. The chunk length decides:

```
bin k → frequency = k × fs / N
```
with fs = 30 Hz (sample rate) and N = 50 (chunk length):

```
bin 0  → 0.0 Hz     a flat line, no wiggle
bin 1  → 0.6 Hz     exactly one full wave across the 50 ticks
bin 2  → 1.2 Hz     two waves
...
bin 25 → 15.0 Hz    up-down every 2 ticks — the fastest thing possible
```

**Why 0.6 Hz is the slowest?** Because the chunk is 50/30 = 1.667 seconds long.
The slowest wiggle you can even see is one that completes exactly one cycle in
that time: 1/1.667 = 0.6 Hz. Anything slower just looks like a trend.

**Why 15 Hz is the fastest?** You need at least 2 samples to see one up-and-down.
With 30 samples per second, that's 15 cycles per second maximum. This is called
the **Nyquist limit**. Anything faster than 15 Hz cannot be represented at all.

So you get 26 bins (0 through 25). That's it. Not a choice — a consequence of
having 50 samples at 30 Hz.

### What the FFT computes for each bin

For each candidate frequency, it asks: **"how much does this signal look like a
wave at this speed?"** — by multiplying the signal by that wave and adding up.

Worked example on a real elbow chunk, bin k=3 (1.8 Hz):

```
multiply the signal by a COSINE at 1.8 Hz, sum it  →  −9.7939
multiply the signal by a SINE   at 1.8 Hz, sum it  →  +12.4895
power = (−9.7939)² + (12.4895)²                    =  251.91

numpy's rfft for that bin                          =  251.91   ← identical
```

That's the whole operation. If the signal genuinely wiggles at 1.8 Hz, it lines
up with the 1.8 Hz wave and the products add up big. If it doesn't, positives
and negatives cancel out and you get ~0.

**Why both sine and cosine?** A wave might start at a peak, or start at zero, or
anywhere between. Cosine catches the first kind, sine the second. Squaring both
and adding makes the result **blind to where the wave happens to start** — you
get "how much 1.8 Hz is in here" regardless of its timing.

The number you get is called **power**. It's an energy-like quantity: bigger
means more of that frequency.

### Removing the average first

Before any of this, we subtract the mean:

```
x = angles − average(angles)
```

Why: if the elbow sits at 96.7° and jiggles by 0.3°, the number 96.7 is enormous
compared to the jiggle. It would land entirely in bin 0 (the "flat" bin) and
dwarf everything. We care about the wiggling, not where the joint is parked. So
subtract the average and only the wiggle remains.

(In signal processing the constant part is called "DC", from direct current.
That's all "DC removed" means: we subtracted the average.)

---

<a name="7"></a>
## 7. The window function

**This is a completely different thing from "the 50-tick chunk", despite both
being called a window. Sorry.**

### What it literally is

`np.hanning(50)` is a list of 50 multipliers:

```
tick  0.. 4 : 0.000 0.004 0.016 0.037 0.064
tick  5.. 9 : 0.099 0.141 0.188 0.241 0.298
tick 10..14 : 0.358 0.420 0.484 0.548 0.611
tick 15..19 : 0.673 0.731 0.786 0.836 0.881
tick 20..24 : 0.919 0.950 0.975 0.991 0.999
tick 25..29 : 0.999 0.991 0.975 0.950 0.919
tick 30..34 : 0.881 0.836 0.786 0.731 0.673
tick 35..39 : 0.611 0.548 0.484 0.420 0.358
tick 40..44 : 0.298 0.241 0.188 0.141 0.099
tick 45..49 : 0.064 0.037 0.016 0.004 0.000
```

Starts at 0, rises to 1 in the middle, back to 0. You multiply your data by it,
element by element. Ends get squashed, middle survives.

### Why it is needed — the actual problem

**The FFT does not analyze your 50 numbers once. It assumes they repeat
forever.**

Take a perfectly smooth rotation, no shaking at all:

```
0.0  0.2  0.4  0.6  ...  9.6  9.8  10.0
```

Here is what the FFT actually sees, because it loops:

```
... 9.4  9.6  9.8  10.0  0.0  0.2  0.4 ...
                    ↑
              a 10° CLIFF
```

**That cliff does not exist on your robot.** The arm never jumped 10° in one
tick. It is an artifact of gluing the end back onto the start.

Mathematically, a sudden cliff contains every frequency at once. So the FFT
reports a huge amount of high-frequency content in a signal that has none.

Measured on that perfectly smooth ramp:

| | fracE_bw |
|---|---|
| **no window** | **0.2391** — claims 24% of a smooth ramp is buzz. A lie. |
| **with hann** | **0.0044** — 0.4%. Correct. |

The window squashes both ends to zero, so end and start match, the cliff
vanishes, and the fake buzz vanishes with it.

### The cost — this is not free

Squashing the ends means partly ignoring real data there. On a real elbow chunk
that had large genuine motion at its edges, windowing moved fracE_bw the *other*
way: 0.213 → 0.315, because it suppressed real low-frequency content.

So the window is a trade: it removes fake high frequency but discounts real data
near the boundaries.

### Does the choice matter?

Tested on the real data:

| window | SmolVLA | pi0.5 | ratio | AUC |
|---|---|---|---|---|
| none | 0.3399 | 0.1961 | 1.7× | 0.923 |
| hann | 0.2973 | 0.0616 | 4.8× | 0.978 |
| hamming | 0.2883 | 0.0601 | 4.8× | 0.978 |
| blackman | 0.3106 | 0.0690 | 4.5× | 0.974 |

Any tapered window gives essentially the same answer. **The absolute number
shifts; the verdict never does.** Rule: use the same window for everything you
compare.

---

<a name="8"></a>
## 8. fracE_bw — energy the motor cannot follow

### The idea

Your motor is a low-pass filter. It can follow slow commands. Fast commands it
physically cannot execute — that energy becomes heat and vibration instead of
motion.

So: **what fraction of the command is too fast to become movement?**

### Where the cutoff comes from — measured, not guessed

We identified a servo model from the ramp test (section 12). It has a
proportional gain `kp ≈ 7.8–9.6`. A first-order system with gain kp has a corner
frequency of `kp / 2π`:

| joint | f_bw |
|---|---|
| shoulder_pan | 1.27 Hz |
| shoulder_lift | 1.43 Hz |
| elbow_flex | 1.40 Hz |
| wrist_flex | 1.25 Hz |
| wrist_roll | 1.30 Hz |
| gripper | 1.52 Hz |

**Your servo's bandwidth is about 1.3 Hz.** That is slow. Its time constant is
1/kp ≈ 125 ms.

### The computation, per joint

```
1. x = angles − mean(angles)              remove the average
2. xw = x × hann_window                   fade the ends
3. P = |FFT(xw)|²                         26 power values, one per bin
4. fracE_bw = sum(P where f > f_bw) / sum(P)
```

### Fully worked example — real elbow chunk

The 26 bins:

```
   0.0 Hz   15.2%  ##################                  SLOW (motor can follow)
   0.6 Hz   49.8%  ###################################  SLOW
   1.2 Hz    3.6%  ####                                 SLOW
  ──────────── cutoff 1.3 Hz ────────────
   1.8 Hz    5.4%  ######                               too fast
   2.4 Hz    2.1%  ##
   3.0 Hz    3.2%  ###
   3.6 Hz    0.4%
   4.2 Hz    1.7%  ##
   4.8 Hz    3.7%  ####
   ... 17 more bins, all too fast
  15.0 Hz    1.3%  #
```

The big bar at 0.6 Hz is the real work — the elbow sweeping through the task.
Everything below the line is buzz.

```
sum of fast bins  = 1479.7
sum of all bins   = 4697.3
fracE_bw(elbow)   = 1479.7 / 4697.3 = 0.3150  → 31.5%
```

### All six joints, one real chunk of each policy

| motor | cutoff | SmolVLA | pi0.5 |
|---|---|---|---|
| shoulder_pan | 1.27 Hz | 6.2% | 0.7% |
| shoulder_lift | 1.43 Hz | 24.2% | 3.1% |
| elbow_flex | 1.40 Hz | 31.5% | 0.6% |
| wrist_flex | 1.25 Hz | 18.0% | 1.1% |
| wrist_roll | 1.30 Hz | 26.2% | 0.7% |
| gripper | 1.52 Hz | 57.0% | 4.2% |
| **average** | | **27.2%** | **1.7%** |

### Overall measured values

| | fracE_bw |
|---|---|
| human demos | 7.7% |
| pi0.5 | 6.2% |
| SmolVLA | 30–36% |

**A third of SmolVLA's command energy is at frequencies the motor cannot
produce.** That is the shaking, quantified.

### Robustness to the cutoff choice

| cutoff | SmolVLA | pi0.5 | ratio | AUC |
|---|---|---|---|---|
| 0.8 Hz | 0.4569 | 0.1963 | 2.3× | 0.917 |
| 1.3 Hz | 0.3288 | 0.0616 | 5.3× | 0.972 |
| 1.8 Hz | 0.2673 | 0.0342 | 7.8× | 0.985 |
| 3.0 Hz | 0.2117 | 0.0212 | 10.0× | 0.982 |
| 5.0 Hz | 0.1630 | 0.0150 | 10.9× | 0.982 |

You could be wrong about the bandwidth by 4× and get the same verdict. This is
the one frequency-domain term that does not inherit the "guessed constant"
disease.

### A quirk you must know

Bins land at 0, 0.6, 1.2, 1.8… A 1.3 Hz cutoff **cannot actually be applied at
1.3 Hz** — the last excluded bin is 1.2, the first included is 1.8. Your
effective cutoff is 1.8 Hz. (Proof: cutoffs of 0.8 and 1.0 Hz give byte-identical
results, because both fall between the same two bins.)

### Why fracE_bw generalizes when R does not

fracE_bw is normalized by the **actuator's measured bandwidth**. Reversing at
10 Hz is fine if your motors do 10 Hz, pathological if they do 1.3 Hz. That is
the physically correct question and it asks it automatically on any robot.

---

<a name="9"></a>
## 9. p99_v and p99_a — how fast, how hard

### The formulas

```
p99_v = 99th percentile of |d1 / dt|      over all ticks and joints
p99_a = 99th percentile of |d2 / dt²|
```

### What "99th percentile" means

Sort all the values from smallest to largest. The 99th percentile is the value
99% of the way up. It's "almost the maximum" but ignores the single most extreme
value, so one corrupt sample can't ruin it.

For a 50-frame chunk: d2 has 48 rows × 6 joints = 288 values, so p99 is roughly
the **3rd largest**. That's very close to a max — which is deliberate.

### Why not the mean

Because we want to catch **violence**, not typical behaviour. A chunk that is
calm for 49 ticks and vicious for one is dangerous, and a mean would hide it.

### Measured values

| | p99_v (rad/s) | p99_a (rad/s²) |
|---|---|---|
| demos | 2.128 | **16.72** |
| pi0.5 | 1.824 | 22.76 |
| SmolVLA | 2.644 | **110.13** |

p99_a is the **strongest single number in the whole analysis**: SmolVLA is
**+10.1 standard deviations** above the human demo mean. It commands
accelerations 6.6× anything a human ever demonstrated on this arm.

### This is what catches isolated violence

Injecting one reversal into a clean human chunk:

| spike | R (blind) | p99_a (catches it) | flagged? |
|---|---|---|---|
| 0° | 0.245 | 18.0 | pass |
| 2° | 0.265 | 32.9 | pass |
| **4°** | 0.283 | **64.3** | **FLAGGED** |
| 25° | 0.393 | 394.1 | FLAGGED |

**R and p99_a are complementary.** R catches sustained chattering; p99_a catches
one-off snaps. They correlate only −0.63, so neither replaces the other.

### Known weakness

p99 over 288 values ≈ 3rd largest, which is why it catches single spikes. **If
you widen the chunk, p99 starts diluting the same way R does.** For guaranteed
detection of isolated events at any window length, use `max` instead of `p99`.

---

<a name="10"></a>
## 10. min_invk — how close to a singularity

### What a singularity is

The arm's joints move; the gripper moves as a result. Usually a small joint
movement causes a small gripper movement. But in certain poses — arm fully
stretched out, or joints lined up — the arm **loses the ability to move the
gripper in some direction at all**. To move even slightly that way, the joints
would have to move enormously, or infinitely.

Those poses are called **singularities**. Near them the arm is fragile: tiny
command errors cause huge joint motions, and controllers go unstable.

### The Jacobian

The **Jacobian** J is the table that says "if I move each joint a little, how
does the gripper move?" It has one row per gripper direction (3 for position,
3 for rotation = 6 rows) and one column per joint.

### THE CRITICAL CORRECTION FOR THIS ARM

The textbook singularity measure is Yoshikawa's manipulability:

```
m = √det(J Jᵀ)
```

**On the SO-101 this is always exactly zero and cannot be used.**

Why: the arm has 6 motors, but the **Jaw only opens the gripper — it does not
move the gripper's position or orientation at all**. Its Jacobian column is
exactly zero. So only 5 joints actually position the gripper, and the effective
Jacobian is 6 rows × 5 columns. A 6×6 matrix built from that (`J Jᵀ`) can never
have full rank, so its determinant is 0.

Measured at the neutral pose: `det(J Jᵀ) = 6.6e-21`. That's zero plus floating
point noise.

**The correct form for an arm with fewer joints than task dimensions:**

```
m = √det(Jᵀ J) = product of the singular values
```
computed on the 6×5 block, dropping the Jaw column. Measured at neutral: 0.0155.
A real, usable number.

### Singular values and the condition number

Any matrix can be decomposed into **singular values** — think of them as "how
much the arm can move the gripper along each of 5 independent directions".

- **σ_max** = the easiest direction
- **σ_min** = the hardest direction

```
1/κ = σ_min / σ_max        the "inverse condition number", between 0 and 1
```

- 1/κ near 1: the arm moves equally well in all directions. Healthy.
- 1/κ near 0: there is a direction the arm can barely move in. Near singular.

We take the **minimum over the chunk** — the worst pose it passes through.

### Measured values

| | min 1/κ | min σ_min |
|---|---|---|
| SmolVLA r1 | **0.00001** | **0.00003** |
| SmolVLA r2 | 0.00001 | 0.00003 |
| pi0.5 | 0.01213 | 0.02107 |

For scale, sampling 4000 random poses across the whole workspace:
median 1/κ = 0.0265, 1st percentile = 0.00039.

**SmolVLA drives the arm to 1/κ = 0.00001 — worse than the worst 1% of its
entire workspace. Essentially into a singularity.** pi0.5 stays near the
workspace median.

AUC 0.024 (translational), 0.037 (full) — near-perfect separation, and via a
completely different mechanism than R or fracE_bw.

### Cost

One SVD of a 6×5 matrix per waypoint. Microseconds. Free.

---

<a name="11"></a>
## 11. seam_v — the join between chunks

### The problem

The policy outputs chunk after chunk. The last action of one chunk and the first
action of the next were computed **independently**. Nothing forces them to line
up. So there can be a jump at the join — a "seam".

### The formula

```
seam_v = max over joints of  |a_first_of_new_chunk − a_last_of_previous| / dt / q̇_peak
```

That is: the size of the jump, converted to an implied velocity, expressed as a
fraction of what the joint can actually do.

### THE VERSION WE REJECTED, AND WHY

The original S_cont normalized by the policy's own typical step:

```
S_cont = gap / mean|d1|        ← WRONG
```

Measured with that formula:

| | S_cont | absolute gap | mean\|d1\| |
|---|---|---|---|
| SmolVLA | 3.52 | 0.0649 rad | 0.0187 |
| pi0.5 | **9.49** | 0.0660 rad | **0.0099** |

The **absolute gaps are nearly identical** (0.0649 vs 0.0660 rad). The entire
difference is the denominator: pi0.5 takes half-sized steps because it is
smooth, so dividing by its own step size inflates its score 2.7×.

**Dividing by the policy's own statistics punishes it for being smooth.** This
made Φ rank the visibly-smooth policy as worse than the visibly-shaking one.

**General rule: never normalize a term by a quantity the thing being scored
controls.** Normalize by hardware limits or by demonstrations.

### The measured seam problem is real in both policies

Implied seam velocity vs the measured limit:

| | seam velocity | vs q̇_max | seams over limit |
|---|---|---|---|
| SmolVLA r1 | 4.46 rad/s | 1.23× | 50% |
| SmolVLA r2 | 3.55 rad/s | 1.02× | 44% |
| pi0.5 | 4.92 rad/s | 1.46× | **80%** |

Both policies demand impossible velocities at chunk joins. pi0.5 is actually
*worse*. **The seam does not distinguish the policies — it is a shared defect.**

---

<a name="12"></a>
## 12. The gates: S_pos, S_vel, S_env

Gates are hard yes/no checks: does this ask for something impossible?

```
n_pos = count of waypoints where |angle| > calibrated range
n_vel = count where |velocity| > measured peak speed
n_env = count where torque exceeds what the motor can make at that speed
```

### Where the limits came from — the ramp test

The original Φ used **guessed** limits, and that is why it failed. Every guessed
limit is either too high (term never fires) or too low (term swamps everything).
There is no safe middle when guessing.

So we measured them. The ramp test drove each joint to saturation and logged
at 1.7 kHz:

| joint | q̇ peak (rad/s) | q̇ sustained | q̈ accel (rad/s²) |
|---|---|---|---|
| shoulder_pan | 4.52 / 4.12 | 3.87 / 3.34 | 30.7 / 15.6 |
| shoulder_lift | **5.31 / 4.10** | 4.19 / 3.56 | 26.2 / 22.8 |
| elbow_flex | 4.85 / 4.38 | 4.03 / 3.88 | 26.2 / 28.1 |
| wrist_flex | 4.53 / 4.40 | 3.78 / 3.49 | 27.7 / 25.7 |
| wrist_roll | 4.51 / 4.59 | 3.87 / 3.77 | 27.7 / 30.7 |
| gripper | 2.64 / 2.68 | 2.49 / 2.48 | 24.8 / 29.3 |

(two numbers = the two directions of travel)

Note shoulder_lift: **5.31 rad/s one way, 4.10 the other.** That's gravity —
it's the joint carrying the arm's weight.

Each number was verified two independent ways: differentiating the logged
position, and reading the servo's own velocity register. They agreed to 2–4%.

### The servo model, from the same data

Fitting the step responses gave a **rate-limited** model (not the second-order
model the literature suggests):

```
velocity  saturates at v_max
acceleration saturates at a_max
tracking a delayed target with gain kp
```

| | identification error | held-out validation error |
|---|---|---|
| second-order + delay | 2.12–3.55° | 3.25–20.8° |
| **rate-limited** | **0.43–1.62°** | **0.65–2.93°** |

The rate-limited model wins on **every joint**. The tell that the second-order
model is wrong: its fitted dead time came out as **927 ms and 1443 ms** — the
optimizer wandering because no linear structure fits.

This model is where `kp` (hence f_bw for fracE_bw) and the 125 ms time constant
come from.

### A gate must use the PEAK, not the sustained value

A gate asks "can the motor **ever** do this?" — so it must use the peak.

We first used the sustained plateau, and the gate **rejected 35% of human
demonstrations**. That is a miscalibrated gate, not a bad demo. Switching to
peak cut demo violations by 68%.

### Why the gates are still not absolute

Even with peak limits, **24% of human demonstrations still trip a gate.**

The reason is structural: the dataset's `action` field is the **leader arm's**
position during teleoperation. The leader is not bound by the follower's
dynamics — a human can move the leader faster than the follower can track. Same
relationship as policy `cmd_` vs achieved `pos_`.

**Commanded infeasibility is normal, even in demonstrations that worked
perfectly.** So gates on commands should be scored as **rates relative to
demos**, not pass/fail:

| gate | demo rate | SmolVLA | pi0.5 |
|---|---|---|---|
| S_env | 0.103 /window | 1.15 (11× demo) | 0.00 (0×) |
| S_vel | 0.468 /window | 1.00 (2.1×) | 0.375 (0.8×) |
| S_pos | 1.005 /window | 0.03 (0.03×) | 0.25 (0.25×) |

### S_env — and why we now recommend dropping it

The idea was good: a motor cannot deliver stall torque and top speed at the same
time. Available torque falls with speed:

```
τ_avail(ω) = τ_stall × (1 − |ω| / ω_free)
```

| ω (rad/s) | 0 | 1 | 2 | 3 | 3.5 | 4 |
|---|---|---|---|---|---|---|
| τ_avail (N·m) | 2.94 | 2.32 | 1.69 | 1.07 | 0.76 | 0.45 |

This *should* catch trajectories that pass both box checks but demand an
impossible (torque, speed) pair. And it appeared to work: S_torque fired 0
times, S_env fired 4–5 times, all in SmolVLA.

**But we tested it and it does not hold up.** Two findings:

1. **Every violation occurs at 77–100% of free speed**, where τ_avail has
   already collapsed to 0.04–0.29 N·m. At those speeds *any* torque estimate
   trips the gate. **It is a velocity check wearing a torque costume.** The
   torque number contributes almost nothing.

2. **The count is estimator-dependent.** Using forward differences (`np.diff`)
   vs central differences (`np.gradient`) on identical data:

   | run | np.diff | np.gradient |
   |---|---|---|
   | SmolVLA r1 | **20** | **2** |
   | SmolVLA r2 | **19** | **4** |

   A 10× difference from the derivative estimator alone. The "11× demo rate"
   figure was inflated by this.

**Recommendation: drop S_env.** It adds nothing S_vel doesn't already catch and
imports two unvalidated constants to do it.

---

<a name="13"></a>
## 13. RNEA — where torque numbers come from

### What it is

RNEA (Recursive Newton-Euler Algorithm) computes: given the arm's pose, its
joint velocities, and its joint accelerations, **what torque must each motor
produce?** It uses the masses, centres of mass, and inertias from the URDF.

### The masses of this arm

| link | mass (kg) | lever arm |
|---|---|---|
| Rotation | 0.100006 | 0.0398 m |
| Pitch | 0.103000 | 0.0921 m |
| Elbow | 0.104000 | 0.0998 m |
| Wrist_Pitch | 0.079000 | 0.0478 m |
| Wrist_Roll | 0.087000 | 0.0252 m |
| Jaw | 0.012000 | 0.0357 m |
| **total** | **0.485 kg** | |

**The whole arm weighs less than half a kilo.**

### Why S_torque was always exactly zero

The original Φ used `TAU_MAX = 3.0 N·m` (the STS3215 datasheet stall torque).
But measured on real recorded motion:

| | peak |τ| | % of the 3.0 limit |
|---|---|---|
| SmolVLA | 0.998 N·m | 33% |
| pi0.5 | 0.666 N·m | 22% |

Worst case over 20,000 random poses: 0.86 N·m gravity torque. Worst case with
the scorer's own placeholder speed/accel limits: 1.77 N·m — still only 59%.

**The arm cannot generate 3 N·m. The term could never fire.** It was like a
speed camera set to 500 mph — it wasn't that trajectories passed, it's that the
camera could not flash.

### The bigger problem: we never measured torque

**There is no torque sensor on this arm.** So:

| quantity | source | status |
|---|---|---|
| τ | RNEA from URDF inertias | **model output, not measured** |
| τ_stall = 2.942 N·m | datasheet | **quoted, never verified** |
| ω_free = 5.40 rad/s | our ramp test | measured ✅ |
| linear envelope shape | back-EMF theory | **assumed** |

Worse, the project notes already record that this route **failed**: measured
servo load correlates with RNEA torque at only r ≈ 0.26, versus r ≈ 0.8 against
velocity. The servo's load reading is dominated by friction, not dynamics.

**Conclusion: torque is effectively unobservable on this arm, and every
torque-based term (S_torque, S_env, S_robust) should be treated as
ungrounded.**

To fix it you would hang known masses at a known lever, hold static poses, and
calibrate the load reading against the gravity torque RNEA predicts — gravity
being the one part of RNEA that is trustworthy, since it's just mass × lever
with no friction or inertia involved.

---

<a name="14"></a>
## 14. D_demo and Mahalanobis distance

### The problem this solves

Every term above compares one policy to another. That tells you which is worse,
but not whether either is **acceptable**.

We need an absolute reference. The demonstrations are it: **200 episodes of a
human teleoperating this exact arm to do this exact task.** Those trajectories
demonstrably work. So the question becomes:

> Is this chunk like the things that are known to work?

### The feature vector

Six numbers per chunk:

```
x = [ R, fracE_bw, p99_v, p99_a, min_invk, seam_v ]
```

From 1515 demo windows we compute the mean μ and the covariance Σ.

### Why not just add the features up

Two reasons.

**Reason 1: the units don't mix.** R is ~1 (unitless). p99_a is ~100 (rad/s²).
min_invk is ~0.02 (a ratio). Add them raw and p99_a *is* the score; nothing else
matters. That is exactly the S_acc disaster — a term 95× everything else purely
because of scale.

**Reason 2: the features overlap.** Adding correlated features double-counts.

### What covariance is

Covariance records **how features move together** across the demonstrations.
Measured on the 1515 demo windows:

| | R | fracE_bw | p99_v | p99_a | min_invk | seam_v |
|---|---|---|---|---|---|---|
| **R** | 1.00 | 0.54 | −0.69 | −0.63 | −0.25 | −0.33 |
| **fracE_bw** | 0.54 | 1.00 | −0.37 | −0.29 | −0.18 | −0.20 |
| **p99_v** | −0.69 | −0.37 | 1.00 | 0.84 | −0.03 | 0.30 |
| **p99_a** | −0.63 | −0.29 | 0.84 | 1.00 | −0.06 | 0.28 |
| **min_invk** | −0.25 | −0.18 | −0.03 | −0.06 | 1.00 | 0.13 |
| **seam_v** | −0.33 | −0.20 | 0.30 | 0.28 | 0.13 | 1.00 |

Read: R and p99_v are **−0.69** — strongly *anti*-correlated. Dithery chunks are
**slow** chunks, which makes physical sense: a joint that keeps reversing never
builds up speed.

The largest correlation is 0.84 (p99_v with p99_a — fast motion needs
acceleration). All below the 0.90 pruning threshold, so nothing was dropped.

### The formula

```
D_demo = √( (x − μ)ᵀ Σ⁻¹ (x − μ) )
```

Plain reading: **how many "demo standard deviations" away is this chunk, after
accounting for the fact that the features overlap?**

If Σ were the identity (no correlations, unit variances), this reduces to plain
Euclidean distance from the demo mean. The Σ⁻¹ is what makes it smart: it
divides out shared variation so overlapping features stop double-counting, and
it recognises *impossible combinations*.

### Worked example — two real chunks

**SmolVLA, chunk #3:**

| feature | value | demo mean | z-score |
|---|---|---|---|
| R | 1.4721 | 0.6194 | +2.17 |
| fracE_bw | 0.2718 | 0.0766 | +3.17 |
| p99_v | 2.6509 | 2.1280 | +0.40 |
| p99_a | 143.08 | 16.72 | **+13.66** |
| min_invk | 0.0165 | 0.0223 | −0.39 |
| seam_v | 0.6595 | 0.1959 | +2.01 |

naive sum of |z| = 21.80 → **Mahalanobis = 25.86 (HIGHER)**

Why higher? Because in the demos R and p99_a are **negatively** correlated
(−0.63). SmolVLA has *both* high R *and* enormous acceleration. That is not just
unusual — it is a combination the demonstrations say should not happen together.
Mahalanobis sees the impossible combination and penalises it beyond the sum of
its parts.

**pi0.5, chunk #3:**

z = [−0.51, −0.96, −0.22, −0.39, −0.47, **+5.17**]

naive sum = 7.73 → **Mahalanobis = 5.77 (LOWER)**

Why lower? Its only outlier is the seam. The other five sit slightly *below* the
demo mean. A naive sum adds those five as if being calmer than a human were a
fault (2.56 of the 7.73). Mahalanobis doesn't.

### The threshold

The demos' own D values have a 95th percentile of **3.98**. That is the pass
mark: a chunk inside 3.98 looks like something that works.

| | mean D | % of chunks outside the envelope |
|---|---|---|
| demos | 2.07 | 5% (by construction) |
| pi0.5 | 4.25 | 56% |
| SmolVLA | 18.2–19.0 | **100%** |

---

<a name="15"></a>
## 15. The final Φ, assembled

```
Φ(chunk) = 10 × n_gate_violations  +  D_demo
```

Only **two** parts. The six features are **inputs to D_demo**, not separate
terms.

The weight of 10 makes any single hard violation outrank any amount of graded
roughness.

### Results on real data

| run | chunks | S_pos | S_vel | S_env | D_demo | **Φ** | % feasible | % inside envelope |
|---|---|---|---|---|---|---|---|---|
| **DEMOS** | 1515 | 1522 | 709 | 156 | 2.07 | **2.25** | 76% | 95% |
| SmolVLA r1 | 17 | 1 | 16 | 20 | 18.19 | **26.73** | 35% | 0% |
| SmolVLA r2 | 17 | 0 | 18 | 19 | 19.00 | **28.82** | 35% | 0% |
| pi0.5 | 16 | 4 | 6 | 0 | 4.25 | **6.38** | 62% | 44% |

**Ordering: demos 2.25 < pi0.5 6.38 < SmolVLA 26.7–28.8.** Matches what you can
see on the hardware, with human demonstrations correctly scoring best.
Separation AUC 0.879, p = 1.9e-05.

### Validation on a held-out demonstration

Episode 160 of the dataset, scored in 7 windows:

```
 win     R  fracE_bw  p99_a  min_invk  gates     D   phi
   0 1.731     0.153   4.14     0.008      0  2.920 2.920
   1 1.128     0.156  12.61     0.008      0  2.192 2.192
   2 0.245     0.117  17.95     0.017      0  2.250 2.250
   3 0.671     0.173  13.44     0.043      0  2.458 2.458
   4 0.412     0.065  30.38     0.043      0  2.703 2.703
   5 0.829     0.062   9.13     0.043      0  1.893 1.893
   6 0.349     0.082  30.56     0.008      0  3.009 3.009
```

**7/7 windows pass. Zero gate violations. Mean Φ = 2.49.** Φ does not
false-positive on known-good motion.

**Look at window 0: R = 1.731** — higher than SmolVLA's average of 1.54 — and it
passes, because its acceleration is only 4.1 rad/s². **This is the key lesson:
fast direction change is not itself bad. Fast direction change at high
acceleration is bad.** A human hovering gently reverses often, softly, and
that's fine.

---

<a name="16"></a>
## 16. Things we tried and threw away

Recording these matters as much as the ones that worked.

| term | result | why rejected |
|---|---|---|
| **S_torque (box)** | always exactly 0 | limit 3.0 N·m, arm's peak is 1.0. Cannot fire. |
| **S_acc** | partial r ≈ +0.01 | q̈_max was a guess wrong by ~10×; penalty was 95× everything else |
| **S_cont = gap/step** | ranked policies **backwards** | dividing by the policy's own step size punishes smoothness |
| **λ_S (SPARC)** | AUC 0.24 at best | collapsed under velocity control; needed high-pass + mean-speed controls; and it moved *opposite* to jerk on real data |
| **S_robust (ensemble)** | 0 violations | ±20% inertias + 50 g payload gives worst case 1.9 N·m, still under a 3.0 limit. Inherits S_torque's vacuity. |
| **self-collision** | AUC 0.660 | convex hulls inflate geometry; flags trajectories that demonstrably executed. Needs proper convex decomposition. |
| **cross-joint coherence** | AUC 0.654 | hypothesis was that dither is uncorrelated across joints and real motion is coordinated. **Measured: demos 0.215, pi0.5 0.209, SmolVLA 0.221 — identical. Hypothesis wrong.** |
| **S_env** | see §12 | a velocity test in disguise; count varied 10× with the derivative estimator |

### A note on SPARC specifically

SPARC is the established smoothness metric from the motor-control literature, so
it deserved a fair test. It got one:

- Validated correctly on synthetics (min-jerk bell scored −1.400 against a −1.4
  target; amplitude and duration invariance held to 4 decimal places).
- On real data its discrimination was AUC 0.24 only after high-pass filtering,
  and it **collapsed to chance** (0.470, p = 0.70) once window mean speed was
  controlled for.
- Most damningly: on the TOPP-RA re-timed execution, physical jerk **improved
  28%** while SPARC said the motion got **worse** (−6.14 → −8.28). The two
  measures moved in opposite directions on the same physical motion.

---

<a name="17"></a>
## 17. Known problems — read before trusting any of this

Ordered by how much they should worry you.

**1. n = 3.** Two SmolVLA runs and one pi0.5 run. Every AUC in this document
rests on essentially one good policy and one bad one. AUC 0.879 for Φ is
computed from 50 chunks total. This is enough to justify design decisions; it is
nowhere near enough to publish.

**2. Torque is unobservable on this arm.** No torque sensor; RNEA is unvalidated
(r ≈ 0.26 against measured load); τ_stall is a datasheet number. Every
torque-derived term is ungrounded. See §13.

**3. S_env should be removed.** Velocity test in disguise, and estimator-
dependent by 10×. See §12.

**4. The gates are not absolute.** 24% of human demonstrations still trip one.
They must be read as rates relative to demos, not pass/fail.

**5. p99 dilutes at longer windows.** It works here because 288 samples makes
p99 ≈ 3rd largest. Widen the chunk and it degenerates toward a mean. Use `max`
if you change the horizon.

**6. Self-collision uses convex hulls**, which inflate the geometry and produce
false positives on trajectories that physically executed. Needs proper convex
decomposition.

**7. Episode 160 is in the demo set** used to build μ and Σ, so scoring it
against that distribution is mildly circular. With 200 episodes the effect is
about 1/200, but it is not a clean held-out test.

**8. R and reversal rate do not generalize across tasks.** Both would wrongly
condemn a balancing controller that legitimately reverses fast. Only fracE_bw
(normalized by measured actuator bandwidth) and D_demo (referenced to demos of
the same task) transfer safely.

**9. Two files still carry a wrong label.** `sparc_compare.py` and
`Research_notes_sparc.txt` call `record_run1` "ACT". It is **SmolVLA** —
confirmed by md5 match to `pvd_logs/record_run1.csv`, which
`validate_spos_sacc.py:23` tags `"smolvla"`. The string "ACT" appears nowhere in
the repo outside files written during this analysis.

---

## Appendix: the measured constants, in one place

```
dt            = 1/30 s = 33.4 ms
encoder tick  = 0.088 degrees (4096 per revolution)
ω_free        = 5.40 rad/s   (measured; datasheet says 4.712)
τ_stall       = 2.942 N·m body, 1.471 gripper  (DATASHEET, unverified)
arm mass      = 0.485 kg total
servo         = rate-limited, kp ≈ 7.8–9.6, time constant ≈ 125 ms, delay 11–14 ms
```

| joint | q̇_peak (rad/s) | q̇_sustained | q_max (rad) | f_bw (Hz) |
|---|---|---|---|---|
| shoulder_pan | 4.52 | 3.34 | 1.93 | 1.27 |
| shoulder_lift | 5.31 | 3.56 | 1.84 | 1.43 |
| elbow_flex | 4.85 | 3.88 | 1.72 | 1.40 |
| wrist_flex | 4.53 | 3.49 | 1.91 | 1.25 |
| wrist_roll | 4.59 | 3.77 | 3.14 | 1.30 |
| gripper | 2.68 | 2.48 | 1.92 | 1.52 |

Demo reference distribution (1515 windows, 200 teleop episodes):

| feature | mean | sd | p95 |
|---|---|---|---|
| R | 0.619 | 0.394 | 1.565 |
| fracE_bw | 0.077 | 0.062 | 0.193 |
| p99_v | 2.128 | 1.311 | 4.333 |
| p99_a | 16.72 | 9.25 | 34.88 |
| min_invk | 0.0223 | 0.0147 | — |
| reversal rate | 0.0103 | — | — |
| **D_demo pass threshold** | | | **3.98** |

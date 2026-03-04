Yes, you are exactly right. The input parsing is split into two distinct parts:

1. **The Graph Data:** A rigid, 3-file format (`meta.dat`, `node.dat`, `link.dat`).
2. **The Rules:** A custom **bytecode-like** format where space-separated integers act as "opcodes" to build and execute path queries.

Here is the breakdown of how to write the files so the code can parse them.

---

### 1. The Graph Input (The 3 Files)

The class `HeterGraph` (in `hin.cpp`) expects a folder path containing exactly these three files.

#### **File 1: `meta.dat**`

This file is used **only** to get the total number of nodes (vertex number).

* **Parsing Logic:** The code reads the first line and calls `stoi(line.substr(17))`.
* **Format:** The first 17 characters are ignored. You must pad the start of the line.
* **Example Content:**
```text
Total graph nodes: 5000

```


*(The parser ignores "Total graph nodes: " and reads "5000".)*

#### **File 2: `node.dat**`

This defines nodes and their types.

* **Parsing Logic:** Reads line-by-line. Splits by tabs `\t` to find the 3rd column, then splits that by comma `,`.
* **Format:** `NodeID` `\t` `NodeName` `\t` `TypeID,TypeID,...`
* **Example Content:**
```text
0	Alice	1
1	PaperA	2
2	Bob	1

```



#### **File 3: `link.dat**`

This defines the edges.

* **Parsing Logic:** Reads line-by-line. Splits by tabs `\t`.
* **Format:** `SourceID` `\t` `TargetID` `\t` `EdgeType,EdgeType...`
* **Example Content:**
```text
0	1	10
2	1	10

```


*(Means Node 0 connects to Node 1 with edge type 10).*

---

### 2. The Rules File Format

This is the confusing part. The file is a single line (or multiple lines) of **space-separated integers**. It works like a **Stack Machine**. The integers serve as commands (Opcodes) or Data.

The parser reads token by token. Here is the dictionary to decode it:

#### **The "Opcodes" (Control Integers)**

| Integer | Meaning | Action in Code |
| --- | --- | --- |
| **-2** | **Forward Edge** | Adds `1` to the direction stack (`->`). |
| **-3** | **Backward Edge** | Adds `-1` to the direction stack (`<-`). |
| **-1** | **Variable Rule** | Sets internal `state = 0`. This signals that the **next** integer read will be the final edge type of a generic query. |
| **-5** | **Instance Rule** | Sets internal `state = 1`. This signals that the **next** integer read will be a specific Node ID to query. |
| **-4** | **Pop / End** | "Backtracks" or cleans up. Removes the last Edge Type and Direction from the stack. |
| **> 0** | **Data** | Interpreted as an Edge Type ID (usually), unless a state flag (`-1` or `-5`) is active. |

---

#### **How to Write a Rule File**

To write a rule file, you must simulate the traversal. The C++ code **executes the query** the moment it finishes reading the data for a `-1` or `-5` state.

**Scenario A: The "Variable" Rule (Generic Query)**
*Query:* Find nodes connected via `Type1 -> Type2 <-`.

* **Step 1:** Define first edge (`Type 1` forward).
* Write: `-2 1` (Direction Forward, EdgeType 1).


* **Step 2:** Define second edge (`Type 2` backward) **AND EXECUTE**.
* To trigger execution, we prepend the specific edge type with `-1`.
* Write: `-3 -1 2` (Direction Backward, **Trigger Variable Mode**, EdgeType 2).
* *At this exact moment, the code runs the algorithm.*


* **Step 3:** Cleanup.
* We added 2 edges to the stack, so we need two pops to reset for the next rule.
* Write: `-4 -4`


* **Final Line:** `-2 1 -3 -1 2 -4 -4`

**Scenario B: The "Instance" Rule (Specific Node Query)**
*Query:* Find nodes connected to **Node 99** via `Type1 -> Type2 <-`.

* **Step 1:** Define first edge.
* Write: `-2 1`


* **Step 2:** Define second edge.
* Write: `-3 2` (Note: we do NOT use `-1` here because this isn't the trigger yet).


* **Step 3:** Define the Instance Trigger.
* Write: `-5 99` (Trigger Instance Mode, Node ID 99).
* *The code runs the algorithm here.*


* **Step 4:** Cleanup.
* Write: `-4 -4`


* **Final Line:** `-2 1 -3 2 -5 99 -4 -4`

### Summary Checklist for your Input

1. **Directory:** Create a folder (e.g., `my_dataset/`).
2. **Meta:** `my_dataset/meta.dat`  Pad the first 17 chars.
3. **Nodes:** `my_dataset/node.dat`  IDs must match `meta.dat` count.
4. **Links:** `my_dataset/link.dat`  Use integers for Edge Types.
5. **Rules:** Write as a space-separated string using the `-2` (fwd), `-3` (bwd), `-1` (var trigger), `-5` (node trigger), and `-4` (pop) logic.
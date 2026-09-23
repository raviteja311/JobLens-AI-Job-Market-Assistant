# Fine-tuning a 0.5B model to replace an LLM call: what actually happened

*Draft for publication. The plan for this project called the post
"Fine-tuning a 3B model to replace an API call: cost and accuracy". The
model that fit in 7.3GB of RAM was 0.5B, the teacher was a local 8B rather
than an API, and the result was not the one the title promised. Every number
is from `docs/experiments.md` in
[JobLens](https://github.com/raviteja311/JobLens-AI-Job-Market-Assistant).*

---

The pitch for distillation is simple. You have a task an LLM does well but
slowly and at a cost per call. You label a few hundred examples with it,
fine-tune a small model on the labels with LoRA, and serve the small model
for a fraction of the price. The interview version ends with "and it matched
the API model at 10x lower cost".

Mine ends with an 80-line regular expression still in production. Here is
how that happened, because the path there taught me more about fine-tuning
than a success would have.

## The task and the setup

JobLens extracts skills from job postings: given a title and description,
return the list of technologies named in it. The reasoning is shallow, the
output is a JSON list, the vocabulary is nearly closed. If any task should
distil, this one should.

- **Student:** Qwen2.5-0.5B-Instruct with LoRA, rank 16, alpha 32, on
  attention and MLP projections. CPU only: the laptop has a 4GB GPU with no
  matching torch wheel and 7.3GB of RAM, which is why the plan's "3B or
  similar" was not reachable.
- **Teacher:** llama3.1 8B over Ollama. The plan calls for a frontier API
  model; no key was configured. This is a materially weaker teacher and
  everything downstream inherits it.
- **Baseline to beat:** the Phase 2 skill dictionary. Eighty canonical
  skills, an alias table, a case-insensitive regex with word boundaries. It
  costs nothing and runs in microseconds.
- **Data:** 242 postings labelled by the teacher, split 182 train and 60
  test before any correction, so nothing could leak.

## Half the teacher's labels did not survive review

Scoring a student against its own teacher measures imitation, and the
teacher scores 1.0 by construction. So the test labels had to be corrected
first, and that is where the first surprise was.

The dictionary is exhaustive over its own vocabulary: if a posting names
PyTorch anywhere, the regex found it. Therefore any dictionary skill the
teacher returned that the regex did not is a skill the posting does not
contain. That is a proof, not a heuristic, and it needs no human. It caught
a lot:

| skill the teacher invented | postings |
| --- | ---: |
| pytorch | 12 |
| python | 10 |
| aws | 5 |
| javascript | 4 |
| postgresql | 3 |

The 8B model was pattern-matching "this is an engineering job" onto the
skills such jobs usually want. A dictionary cannot fail that way, and nothing
in the training loop would have noticed.

For the 80 terms outside the dictionary's vocabulary I had to read. Forty-one
were real gaps in the taxonomy: ansible, react, prometheus, grafana,
playwright, duckdb, ebpf, webassembly. Thirty-nine were the teacher answering
a different question than it was asked: `production ai infrastructure`,
`homeowner tradeoffs`, `energy-aware scheduling and control`. Those describe
what the job does. Nobody puts them on a CV.

Applying both rules to the training split dropped 300 of 610 teacher labels.
Forty-nine percent. I trained on the corrected labels, because distilling a
teacher that invents PyTorch on one posting in five teaches the student to
do the same.

## The student collapsed to always-empty

First fine-tune, three epochs, validation loss falling nicely. Micro F1 on
the test set: **0.000**. The adapter returned `{"skills": []}` for every one
of 60 postings, including one whose gold labels were ansible, aws,
cloudformation, kubernetes, pulumi, saltstack and terraform. Valid JSON,
correct schema, nothing inside.

The temptation was to retrain immediately with the class balance fixed,
because 56% of the training labels were empty. I made myself run six checks
first, cheapest to most expensive. Five were ruled out. One was the cause.

`SFTConfig(completion_only_loss=True)` was set and had no effect. Printing
the labels tensor for one training example:

```
sequence length      : 316
positions masked -100: 0
positions with loss  : 316
```

Zero masked. The trl docstring says the flag is supported only for
prompt-completion datasets. Mine was conversational, a list of messages, so
the flag was silently ignored. No warning. The training log recorded
`completion_only_loss: true` the entire time.

The completion was about 8 tokens of 316. Roughly 97% of every gradient step
was teaching the model to recite the system prompt, the rules, the schema
block and the job posting back. The schema block contained a literal example,
`{"skills": ["python", "pytorch", "aws"]}`, which also explained why the
untuned base model had been returning exactly that list for a Marketing
Student Assistant posting: it was completing the example, not extracting.

The correct flag for a messages dataset is `assistant_only_loss`. It exists,
defaults to False, and needs generation markers in the chat template, which
Qwen2.5's has. The fix was one line. The guard that makes it stick is a
function that inspects the first batch and raises if fewer than half the
positions are masked, so the flag can never be silently ignored again.

## Masking the loss: 0.000 to 0.383

One variable changed. F1 went from 0.000 to 0.383, with the model still
predicting empty on 85% of postings. Rebalancing the training data to 30%
empty and labelling more postings got to 0.369, which is not better, and
told me the data was not the constraint.

The checkpoint selection also changed. Validation loss had been falling
throughout the collapsed run, so loss was not a usable signal for this task.
A callback now generates on a held-out slice after every epoch and keeps the
checkpoint with the best F1 on actual output.

## Then I found the benchmark could not fail the baseline

Someone asked why the empty rate was so high, and answering it meant reading
how the gold labels were built. They were built as: everything the regex
found, plus the teacher's terms filtered through a hand-reviewed vocabulary.
That vocabulary was, in substance, the regex's own capability spec.

Truth had been defined as things the regex could have found. Gold was a
superset of the regex output on all 60 postings, so the regex could not
produce a false positive, and its precision came back as exactly 1.000. I
had published that as a finding. It was an identity.

The gold set was rebuilt blind. Candidates from both labellers pooled and
shuffled, provenance written to a separate file and joined back only after
every decision was recorded. Decisions per occurrence rather than per term,
because "go" is a language in one posting and a verb in the next. And an
additive pass over all 60 postings for skills neither labeller had proposed.
206 occurrences judged, 116 kept, 90 dropped, 22 added.

One dropped term came from the regex. On a Portuguese posting it matched
`excel` inside `Excelência`, because the word boundaries are ASCII and an
accented letter reads as a boundary. Precision went from 1.000 to 0.986. That
0.014 is a real bug the first gold set had made structurally invisible.

## The final table

Retrained on gold-v2-style labels, two epochs, validation F1 still climbing
(0.573 then 0.602) when the run ended. Scored on 60 postings with 138 gold
skill mentions:

| extractor | micro F1 | precision | recall | ms/call |
| --- | ---: | ---: | ---: | ---: |
| rules (80-line regex) | **0.663** | **0.986** | 0.500 | **5** |
| teacher (llama3.1 8B) | 0.634 | 0.551 | **0.746** | 9,512 |
| tuned-v2 (Qwen 0.5B + LoRA) | 0.485 | 0.516 | 0.457 | 4,201 |
| tuned-balanced | 0.369 | 0.559 | 0.275 | 3,612 |
| tuned-masked | 0.337 | 0.674 | 0.225 | 3,348 |
| base (Qwen 0.5B, no tuning) | 0.104 | 0.094 | 0.116 | 6,046 |
| tuned-v1 (collapsed) | 0.000 | 0.000 | 0.000 | 3,105 |

The regex ships. It is not close on precision, it is 800 times faster than
the student, and it cannot hallucinate. The teacher wins on recall because
it knows technologies outside 80 words, and pays for it with a precision of
0.551. The student is a worse version of the teacher at half the latency.

## What I would tell someone starting this

1. **Correct the teacher before you copy it.** Any independent check on the
   labels, even a regex, finds things the loss curve never will.
2. **Print the labels tensor.** One batch, once. The flag you set may not be
   the flag that applies to your dataset format, and the library will not
   tell you.
3. **Select checkpoints on the metric you ship**, generated from real output.
   Validation loss fell smoothly through a run that produced nothing.
4. **Check whether your benchmark can fail your baseline.** If precision
   comes back as exactly 1.000, that is not a result, it is a definition.
5. **Keep the collapsed row in the table.** It is the most informative line
   in the phase.

## Recorded as not done

No inter-annotator agreement: I judged all 60 postings myself in one sitting,
so a second pass would measure my memory. The tooling for an independent
pass exists and reads only the candidate pool. No API-model teacher and no
3B student, for the hardware and budget reasons above. A third epoch, which
the still-climbing validation F1 suggests would have helped.

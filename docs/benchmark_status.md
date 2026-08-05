# Benchmark Status

This file records only verified benchmark states. It deliberately separates confirmed results from unverified work.

## Verified broad benchmark

A broad benchmark run completed with 73 of 73 cases passing.

## Verified later full run

A later full run produced:

- 202 answered;
- 202 correct;
- 5 failures.

The remaining failing cases were:

- `hrq055`;
- `hrq107`;
- `hrq108`;
- `hrq110`;
- `hrq112`.

The shared failure family was overgeneralized negative inference.

## Work after that run

Targeted changes for the five failures were implemented. Thor connectivity failed before a complete benchmark could verify those changes. Therefore the repository must not claim that the five cases pass or that the full suite is clean until a new completed run proves it.

## Reporting rule

Documentation, commits, and release notes must distinguish these states:

- verified passing result;
- verified failing result;
- implemented but unverified change;
- prediction or architectural hypothesis.

No benchmark number should be updated from partial logs, an interrupted run, or expected behavior.

"""Evaluation of planner LLMs for Fed Agent: test cases, runner, scorer and report.

Runs go through the real Fed Agent pipeline (planner, validators, study session),
never execute a task, and talk to a model through an OpenAI-compatible endpoint.
"""

# Two-minute demo recording checklist

Use `run_voice_chat.py` and your own voice. Keep the recording under two
minutes and show both the spoken input/output and the saved evidence/trace.

1. Start with a short policy question whose answer is in the current return or
   delivery policy. Let the agent speak the answer.
2. Ask a follow-up such as "How long does that take?" so the demo shows that
   one session retains conversation context.
3. Ask for an exception or compensation that requires a human decision. Show
   the agent create a ticket and hand the issue over, or ask an unsupported
   policy question and show an honest uncertainty response.
4. Open the corresponding `turn_*/sources.json` and `trace.json` to show the
   policy section IDs and confirmed tool/ticket result.

Do not edit the agent's answers or claim an action succeeded unless the tool
trace confirms it. Export the screen recording as `demo.mp4` after recording.

This project is an experiment to see how an LLM would interact with a physical environment when given the ability to see, hear, talk, and move. I created an AI Chatbot that ran on a NVIDIA DGX Spark with voice recognition and Text-to-Speech.

It is attached to a FIRST Robotics Swerve Drive Base. The AI can use a camera to relatively safely roam around a space and find people to chat with and then have a conversation with them. It has the ability to impersonate fictional characters with a custom text-to-speech voice and personality.

Everything on the project is local to the robot. The NVIDIA DGX Spark and all other components run without internet connectivity. A local open weight LLM powers the system and a Wikipedia copy is available for the LLM to knowledge search.

The AI takes a photo of its surroundings. It then interprets the image and from that interpretation decides whether to move forward, turn left, turn right, or stop and initiate a conversation. It then sends this action to the swerve drives which carry it out.

If a conversation is initiated, the AI stops moving entirely for the duration of the conversation. The conversation can then be ended by saying "exit conversation mode" where the AI will return to roaming.

In the summer of 2026, I used Qwen 3.6 as the model.

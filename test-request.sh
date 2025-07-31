# This is an example of how to send a request to your API.
#
# IMPORTANT:
# The command below is a single line. Copy the entire command and paste it into your terminal.
# Make sure to replace the placeholder values before running.
#
# On Windows (Command Prompt or PowerShell), you might need to escape the double quotes inside the JSON data.
# If the command below gives you trouble, try this Windows-specific version:
# curl -X POST "https://bajaj-nodejs.vercel.app/hackrx/run" -H "Content-Type: application/json" -d "{\"documents\": \"https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D\", \"questions\": [\"What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?\"]}"

# For Linux/macOS/Git Bash:
curl -X POST 'https://bajaj-nodejs.vercel.app/hackrx/run' -H 'Content-Type: application/json' -d '{"documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D", "questions": ["What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"]}'

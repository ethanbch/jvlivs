import sys

emoji = "😊 👋 🌟 😄"
print("Testing emoji print without reconfiguration:")
try:
    # We simulate CP1252 encoding to see if it raises
    emoji.encode(sys.stdout.encoding or 'cp1252')
    print("Default stdout encoding supports emojis directly:", sys.stdout.encoding)
    print(emoji)
except UnicodeEncodeError as e:
    print("Caught expected UnicodeEncodeError:", e)

print("\nReconfiguring sys.stdout...")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Successfully reconfigured sys.stdout!")
    # Now we print the emoji
    print("Emoji printed successfully:", emoji)
except Exception as e:
    print("Failed to reconfigure:", e)

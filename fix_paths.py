import json
import os

# Read report
with open('data/outputs/report.json', 'r') as f:
    report = json.load(f)

# Fix paths: convert backslashes to forward slashes
for student in report['top_3_least_interactive']:
    if student.get('crop_image'):
        # Replace Windows backslashes with forward slashes
        fixed_path = student['crop_image'].replace(os.sep, '/')
        student['crop_image'] = fixed_path

# Save fixed report
with open('data/outputs/report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("Report paths fixed!")
print("\nUpdated paths:")
for student in report['top_3_least_interactive']:
    print(f"Track {student['track_id']}: {student.get('crop_image', 'NONE')}")

Download LLM Studio, Download Model = llava-v1.5-7b, Run the Server and make sure its reachable at http://127.0.0.1:1234, open command prompt run command (python webcam_terminator_hud_final.py)
once running it will display the person's race, weight, and clothes
Here is a .json schema to add to your llm studio: {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LLaVA_TerminatorAnalysis_Extended",
  "type": "object",
  "properties": {
    "gender": {
      "type": "string",
      "enum": ["male", "female", "non-binary", "unknown"]
    },
    "race": {
      "type": "string",
      "description": "Perceived race or ethnicity (e.g., Hispanic, Black, White, Asian, etc.)"
    },
    "age_estimate": {
      "type": "string",
      "pattern": "^\\d{1,3}(\\s?[-–]\\s?\\d{1,3})?$",
      "description": "Estimated age or age range (e.g., '35', '25–30')"
    },
    "weight_estimate": {
      "type": "string",
      "pattern": "^[~≈]?\\d{2,3}\\s?lbs$",
      "description": "Estimated body weight (e.g., '~180 lbs')"
    },
    "facial_hair": {
      "type": "string",
      "description": "Description of beard, mustache, or clean-shaven"
    },
    "hair": {
      "type": "object",
      "properties": {
        "length": { "type": "string", "enum": ["short", "medium", "long"] },
        "color": { "type": "string" }
      },
      "required": ["length", "color"]
    },
    "clothing": {
      "type": "array",
      "items": { "type": "string" },
      "description": "List of visible clothing items"
    },
    "pose": {
      "type": "string",
      "description": "Body pose or orientation (e.g., 'arms crossed', 'hands in pockets', 'leaning forward')"
    },
    "emotion": {
      "type": "string",
      "enum": ["neutral", "confident", "angry", "happy", "sad", "focused", "intimidating", "surprised"],
      "description": "Detected emotional expression from face or body"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "image_id": {
      "type": "string",
      "description": "Optional hashed reference to image crop"
    }
  },
  "required": ["gender", "race", "age_estimate", "weight_estimate", "facial_hair", "hair", "clothing", "pose", "emotion"]
}

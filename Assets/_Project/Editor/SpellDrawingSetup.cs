#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using Unity.InferenceEngine;
using SpellDrawing.CV;

namespace SpellDrawing.EditorTools
{
    /// <summary>
    /// One-click scene setup so you don't have to hand-wire GameObjects and components.
    /// </summary>
    public static class SpellDrawingSetup
    {
        [MenuItem("Tools/Spell Drawing/Create Spell Caster In Scene")]
        public static void CreateSpellCasterInScene()
        {
            var existing = Object.FindFirstObjectByType<SpellCaster>();
            if (existing != null)
            {
                Selection.activeGameObject = existing.gameObject;
                Debug.Log("A SpellCaster already exists in this scene — selected it instead of creating a new one.");
                return;
            }

            var go = new GameObject("SpellCaster");
            go.AddComponent<LineRenderer>();
            go.AddComponent<StrokeDrawer>();
            go.AddComponent<SpellCaster>();
            go.AddComponent<GestureTemplateRecorder>();

            Undo.RegisterCreatedObjectUndo(go, "Create Spell Caster");
            Selection.activeGameObject = go;

            Debug.Log(
                "Created 'SpellCaster' in the scene.\n" +
                "1) Enter Play Mode and draw one or more strokes with the mouse (e.g. a '+' as two " +
                "separate drags) — press Space to cast, Escape to clear and start over.\n" +
                "2) Right-click the GestureTemplateRecorder component header -> 'Save Last Gesture As " +
                "Template' (set 'Next Spell Name' first). Repeat for each spell shape.\n" +
                "3) Drag the saved templates from Assets/_Project/Spells/Templates into the SpellCaster's " +
                "'Spells' list, and optionally assign an Effect Prefab on each template.");
        }

        [MenuItem("Tools/Spell Drawing/Add Rasterizer Preview")]
        public static void AddRasterizerPreview()
        {
            var caster = Object.FindFirstObjectByType<SpellCaster>();
            if (caster == null)
            {
                Debug.LogWarning("No SpellCaster in the scene yet — run 'Create Spell Caster In Scene' first.");
                return;
            }

            var canvas = Object.FindFirstObjectByType<Canvas>();
            if (canvas == null)
            {
                var canvasGo = new GameObject("Canvas", typeof(Canvas), typeof(CanvasScaler));
                canvas = canvasGo.GetComponent<Canvas>();
                canvas.renderMode = RenderMode.ScreenSpaceOverlay;
                Undo.RegisterCreatedObjectUndo(canvasGo, "Create Canvas");
            }

            var imageGo = new GameObject("StrokeRasterPreview", typeof(RawImage));
            imageGo.transform.SetParent(canvas.transform, false);

            var rt = imageGo.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(1f, 1f);
            rt.anchorMax = new Vector2(1f, 1f);
            rt.pivot = new Vector2(1f, 1f);
            rt.sizeDelta = new Vector2(160f, 160f);
            rt.anchoredPosition = new Vector2(-20f, -20f);

            var rawImage = imageGo.GetComponent<RawImage>();
            rawImage.texture = Texture2D.blackTexture;

            var preview = caster.gameObject.GetComponent<StrokeRasterizerPreview>();
            if (preview == null) preview = caster.gameObject.AddComponent<StrokeRasterizerPreview>();

            var so = new SerializedObject(preview);
            so.FindProperty("previewImage").objectReferenceValue = rawImage;
            so.ApplyModifiedProperties();

            Undo.RegisterCreatedObjectUndo(imageGo, "Create Rasterizer Preview");
            Selection.activeGameObject = imageGo;

            Debug.Log(
                "Added a rasterizer preview panel (top-right corner). Enter Play Mode, draw a gesture " +
                "with the mouse, and press Space — you'll see it rasterized to a bitmap live. Enable " +
                "'Save To Disk' on the StrokeRasterizerPreview component (on the SpellCaster object) if " +
                "you want PNGs written to " + Application.persistentDataPath + ".");
        }

        [MenuItem("Tools/Spell Drawing/ML/Add CNN Classifier")]
        public static void AddCnnClassifier()
        {
            var caster = Object.FindFirstObjectByType<SpellCaster>();
            if (caster == null)
            {
                Debug.LogWarning("No SpellCaster in the scene yet — run 'Create Spell Caster In Scene' first.");
                return;
            }

            var modelGuids = AssetDatabase.FindAssets("spell_classifier t:ModelAsset");
            ModelAsset model = modelGuids.Length > 0
                ? AssetDatabase.LoadAssetAtPath<ModelAsset>(AssetDatabase.GUIDToAssetPath(modelGuids[0]))
                : null;
            var classesAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(
                "Assets/_Project/Models/spell_classifier_classes.json");

            if (model == null || classesAsset == null)
            {
                Debug.LogWarning(
                    "Couldn't find spell_classifier.onnx (as a ModelAsset) or spell_classifier_classes.json " +
                    "under Assets/_Project/Models. If you just added com.unity.ai.inference, let the Editor " +
                    "finish resolving packages and reimporting the .onnx first, then try again.");
                return;
            }

            var classifier = caster.gameObject.GetComponent<CnnSpellClassifier>();
            if (classifier == null) classifier = caster.gameObject.AddComponent<CnnSpellClassifier>();

            var so = new SerializedObject(classifier);
            so.FindProperty("modelAsset").objectReferenceValue = model;
            so.FindProperty("classesJson").objectReferenceValue = classesAsset;
            so.ApplyModifiedProperties();

            Selection.activeGameObject = caster.gameObject;
            Debug.Log(
                "Added CnnSpellClassifier to SpellCaster, wired to spell_classifier.onnx. Enter Play Mode " +
                "and draw a gesture — CNN predictions log alongside SpellCaster's $1 result, side by side.");
        }
    }
}
#endif

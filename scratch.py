# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)

model: HookedTransformer = HookedTransformer.from_pretrained("pythia-6.9b")
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
common_words = [
    # Existing words
    # ("big", "groß", "grand", "grande"),
    ("small", "klein", "petit", "pequeño"),
    ("fast", "schnell", "rapide", "rápido"),
    # ("slow", "langsam", "lent", "lento"),
    ("hot", "heiß", "chaud", "caliente"),
    ("cold", "kalt", "froid", "frío"),
    # ("good", "gut", "bon", "bueno"),
    ("bad", "schlecht", "mauvais", "malo"),
    ("happy", "glücklich", "heureux", "feliz"),
    ("sad", "traurig", "triste", "triste"),
    # ("new", "neu", "nouveau", "nuevo"),
    # ("old", "alt", "vieux", "viejo"),
    ("high", "hoch", "haut", "alto"),
    # ("low", "niedrig", "bas", "bajo"),
    ("near", "nah", "près", "cerca"),
    ("far", "fern", "loin", "lejos"),
    ("easy", "leicht", "facile", "fácil"),
    ("hard", "schwer", "difficile", "difícil"),
    ("soft", "weich", "doux", "suave"),
    # ("loud", "laut", "fort", "fuerte"),
    ("quiet", "leise", "silencieux", "silencioso"),
    # Additional words
    ("young", "jung", "jeune", "joven"),
    ("dark", "dunkel", "sombre", "oscuro"),
    # ("strong", "stark", "fort", "fuerte"),
    ("weak", "schwach", "faible", "débil"),
    ("sweet", "süß", "doux", "dulce"),
    ("sour", "sauer", "acide", "ácido"),
    # ("long", "lang", "long", "largo"),
    # ("short", "kurz", "court", "corto"),
    # ("wide", "breit", "large", "ancho"),
    ("narrow", "schmal", "étroit", "estrecho"),
    ("deep", "tief", "profond", "profundo"),
    ("shallow", "flach", "peu profond", "poco profundo"),
    ("clean", "sauber", "propre", "limpio"),
    # ("dirty", "schmutzig", "sale", "sucio"),
    ("rich", "reich", "riche", "rico"),
    # ("poor", "arm", "pauvre", "pobre"),
    ("full", "voll", "plein", "lleno"),
    # ("empty", "leer", "vide", "vacío"),
    # ("smart", "klug", "intelligent", "inteligente"),
    ("dumb", "dumm", "bête", "tonto"),
    ("wet", "nass", "mouillé", "mojado"),
    # ("dry", "trocken", "sec", "seco"),
    ("smooth", "glatt", "lisse", "suave"),
    ("rough", "rau", "rugueux", "áspero"),
    ("sharp", "scharf", "aigu", "afilado"),
    ("blunt", "stumpf", "épointé", "desafilado"),
    ("heavy", "schwer", "lourd", "pesado"),
    ("light", "leicht", "léger", "ligero"),
    # ("cat", "Katze", "chat", "gato"),
    ("dog", "Hund", "chien", "perro"),
    ("box", "Kiste", "boîte", "caja"),
    # ("car", "Auto", "voiture", "coche"),
    ("tree", "Baum", "arbre", "árbol"),
    ("bird", "Vogel", "oiseau", "pájaro"),
    ("fish", "Fisch", "poisson", "pez"),
    # ("book", "Buch", "livre", "libro"),
    ("door", "Tür", "porte", "puerta"),
    ("pen", "Stift", "stylo", "bolígrafo"),
    ("cup", "Tasse", "tasse", "taza"),
    ("shoe", "Schuh", "chaussure", "zapato"),
    # ("bag", "Tasche", "sac", "bolsa"),
    ("key", "Schlüssel", "clé", "llave"),
    # ("ball", "Ball", "balle", "pelota"),
    # ("hat", "Hut", "chapeau", "sombrero"),
    # ("bed", "Bett", "lit", "cama"),
    ("chair", "Stuhl", "chaise", "silla"),
    ("milk", "Milch", "lait", "leche"),
    ("egg", "Ei", "œuf", "huevo"),
    ("apple", "Apfel", "pomme", "manzana"),
    # ("orange", "Orange", "orange", "naranja"),
    ("water", "Wasser", "eau", "agua"),
    # ("juice", "Saft", "jus", "jugo"),
    # ("bread", "Brot", "pain", "pan"),
    ("cheese", "Käse", "fromage", "queso"),
    # ("salt", "Salz", "sel", "sal"),
    ("sugar", "Zucker", "sucre", "azúcar"),
    ("meat", "Fleisch", "viande", "carne"),
    ("rice", "Reis", "riz", "arroz"),
    ("soap", "Seife", "savon", "jabón"),
    # ("table", "Tisch", "table", "mesa"),
    ("phone", "Telefon", "téléphone", "teléfono"),
    ("clock", "Uhr", "horloge", "reloj"),
    ("lamp", "Lampe", "lampe", "lámpara"),
    ("window", "Fenster", "fenêtre", "ventana"),
    # ("wall", "Wand", "mur", "pared"),
    # ("floor", "Boden", "sol", "suelo"),
    ("roof", "Dach", "toit", "techo"),
    ("garden", "Garten", "jardin", "jardín"),
    # ("kitchen", "Küche", "cuisine", "cocina"),
    ("bathroom", "Badezimmer", "salle de bain", "baño"),
    # ("office", "Büro", "bureau", "oficina"),
    ("school", "Schule", "école", "escuela"),
    # ("hospital", "Krankenhaus", "hôpital", "hospital"),
    # ("bank", "Bank", "banque", "banco"),
    ("store", "Geschäft", "magasin", "tienda"),
    ("beach", "Strand", "plage", "playa"),
    # ("mountain", "Berg", "montagne", "montaña"),
    ("river", "Fluss", "rivière", "río"),
    # ("lake", "See", "lac", "lago"),
    # ("forest", "Wald", "forêt", "bosque"),
    ("island", "Insel", "île", "isla"),
    # ("street", "Straße", "rue", "calle"),
    # ("city", "Stadt", "ville", "ciudad"),
    # ("country", "Land", "pays", "país"),
    # ("world", "Welt", "monde", "mundo"),
    # ("sun", "Sonne", "soleil", "sol"),
    # ("moon", "Mond", "lune", "luna"),
    # ("star", "Stern", "étoile", "estrella"),
    ("sky", "Himmel", "ciel", "cielo")
]
print(pd.Series([word_tuple[0] for word_tuple in common_words]).value_counts().sort_values())
for word_tuple in common_words:
    for i in range(1, 4):
        x = (model.to_str_tokens(" "+word_tuple[i]))
        if len(x) == 2:
            print(x, word_tuple[0])
    # print()
NOUN_INDEX = 33
HALFWAY = len(common_words)//2
print(len(common_words))
# %%
token_list = [model.to_tokens(list((" " + i for i in word_tuple))) for word_tuple in common_words]
[x.shape for x in token_list]
final_index = np.zeros((len(token_list), 4), dtype=np.int64)
for i, word_tuple in enumerate(common_words):
    for j in range(4):
        final_index[i, j] = len(model.to_tokens(" "+word_tuple[j])[0]) - 1
final_index


# %%
residuals_list = []
cache_list = []
for tokens in tqdm.tqdm(token_list):
    _, cache = model.run_with_cache(tokens)
    residuals_list.append(cache.stack_activation("resid_pre"))
    cache_list.append(cache)
final_residuals = torch.stack([resids[:, np.arange(4), final_index[i]] for i, resids in enumerate(residuals_list)])

# %%
ave_residuals = final_residuals.mean(0)
diff_residuals = ave_residuals[:, 1:, :] - ave_residuals[:, 0:1, :]
diff_residuals /= diff_residuals.norm(dim=-1, keepdim=True)
imshow(einops.einsum(diff_residuals, diff_residuals, "layer lang1 d_model, layer lang2 d_model -> layer lang1 lang2")[::3], x=["G", "F", "S"], y=["G", "F", "S"], facet_col=0, aspect="equal", facet_labels=[str(i) for i in np.arange(n_layers)[::3]], title="Cosine Sim of Average Offset of Each Language to English")

# %%
LANGUAGES = ["German", "French", "Spanish"]
# HALFWAY = 30
layer = 6
eng_to_lang = final_residuals[:, layer, 0:1, :] - final_residuals[:, layer, 1:, :]
x = eng_to_lang[:HALFWAY].mean(0)
# histogram(eng_to_lang[HALFWAY:] @ x, nbins=HALFWAY0)

x = x/x.norm(dim=-1, keepdim=True)
y = (eng_to_lang * x).sum(-1)
z = y / eng_to_lang[:].norm(dim=-1)
line(z.T, x=[word[0] for word in common_words], title=f"Layer {layer}", xaxis="Word", yaxis="FVE", line_labels=LANGUAGES)
y = (eng_to_lang * x[0]).sum(-1)
z = y / eng_to_lang[:].norm(dim=-1)
line(z.T, x=[word[0] for word in common_words], title=f"Layer {layer} against German dir", xaxis="Word", yaxis="FVE", line_labels=LANGUAGES)
# for i, z2 in enumerate(z):
#     print(i, z2.item(), common_words[i+HALFWAY][0])
# %%
eng_to_lang = final_residuals[1:, :, 1, :][:, :, None, :] - final_residuals[:-1, :, 1:, :]
lang_diff = eng_to_lang[:10, :, :].mean(0)
lang_diff = lang_diff/lang_diff.norm(dim=-1, keepdim=True)
fve = (eng_to_lang[10:, :, :] * lang_diff[None, :, :]).sum(-1) / eng_to_lang[10:, :, :].norm(dim=-1)
fve.shape

line(fve.mean(0).T, line_labels=["German", "French", "Spanish"], title="Mean FVE", xaxis="Layer", yaxis="FVE")
line(fve.median(0).values.T, line_labels=["German", "French", "Spanish"], title="Median FVE", xaxis="Layer", yaxis="FVE")
# %%
l5_resids = final_residuals[:, 5, :, :]
l5_resids = l5_resids - l5_resids.mean([0, 1])
x = l5_resids[:, 0, :] / l5_resids[:, 0, :].norm(dim=-1, keepdim=True)
imshow(x @ x.T)
# %%
(l5_resids[1:, 0, :] - l5_resids[:-1, 1, :]).norm(dim=-1).mean()
# %%
l16_resids = final_residuals[:, 16, :, :]
l16_resids_cent = l16_resids - l16_resids.mean(0)
x = l16_resids_cent[:, 0, :] / l16_resids_cent[:, 0, :].norm(dim=-1, keepdim=True)
y = l16_resids_cent[:, 3, :] / l16_resids_cent[:, 3, :].norm(dim=-1, keepdim=True)
print((x @ y.T).diag().mean())
imshow(x @ y.T)

# %%
neuron_list = []
for cache in tqdm.tqdm(cache_list):
    # _, cache = model.run_with_cache(tokens)
    neuron_list.append(cache.stack_activation("post"))
    # cache_list.append(cache)
final_neurons = torch.stack([neurons[:, np.arange(4), final_index[i]] for i, neurons in enumerate(neuron_list)])
final_neurons.shape
# %%
final_neuron_med = final_neurons.median(0).values
final_neuron_med_flat = to_numpy(einops.rearrange(final_neuron_med, "layer lang mlp -> lang (layer mlp)"))
neuron_df = nutils.make_neuron_df(n_layers, d_mlp)
neuron_df["English"] = final_neuron_med_flat[0]
neuron_df["German"] = final_neuron_med_flat[1]
neuron_df["French"] = final_neuron_med_flat[2]
neuron_df["Spanish"] = final_neuron_med_flat[3]
neuron_df["ave"] = final_neuron_med_flat.mean(0)
neuron_df["max"] = final_neuron_med_flat.max(0)
neuron_df["diff"] = neuron_df["max"] - neuron_df["ave"]
neuron_df = neuron_df.sort_values("diff", ascending=False)
nutils.show_df(neuron_df.head(50))
# %%
LANGUAGES = ["English", "German", "French", "Spanish"]
layers = neuron_df.head(10).L.values
indices = neuron_df.head(10).N.values
labels = neuron_df.head(10).label.values


acts = to_numpy(final_neurons[:, layers, :, indices])
temp_df = melt(acts)
temp_df["lang"] = [LANGUAGES[i] for i in temp_df["2"]]
temp_df["label"] = [labels[i] for i in temp_df["0"]]
px.box(temp_df, y="value", color="lang", x="label", title="Top 10 Language Neurons")
# %%
LANGUAGES = ["English", "German", "French", "Spanish"]
ind1 = [0, 0, 0, 1, 1, 2]
ind2 = [1, 2, 3, 2, 3, 3]
line_labels = [f"{LANGUAGES[i1]}-{LANGUAGES[i2]}" for i1, i2 in zip(ind1, ind2)]
eng_to_lang = final_residuals[:, :, ind1, :] - final_residuals[:, :, ind2, :]
lang_diff = eng_to_lang[:HALFWAY, :, :].mean(0)
lang_diff = lang_diff/lang_diff.norm(dim=-1, keepdim=True)
fve = (eng_to_lang[HALFWAY:, :, :] * lang_diff[None, :, :]).sum(-1) / eng_to_lang[HALFWAY:, :, :].norm(dim=-1)
fve.shape

line(fve.mean(0).T, line_labels=line_labels, title="Mean FVE", xaxis="Layer", yaxis="FVE")
line(fve.median(0).values.T, line_labels=line_labels, title="Median FVE", xaxis="Layer", yaxis="FVE")

# %%
final_residuals_cent = final_residuals - final_residuals.mean(dim=0)
final_residuals_cent1 = final_residuals_cent[:, :, ind1, :]
final_residuals_cent1 /= final_residuals_cent1.norm(dim=-1, keepdim=True)
final_residuals_cent2 = final_residuals_cent[:, :, ind2, :]
final_residuals_cent2 /= final_residuals_cent2.norm(dim=-1, keepdim=True)
cosine_sim_med = (final_residuals_cent2 * final_residuals_cent1).sum(-1).median(0).values
fig = line(cosine_sim_med.T, line_labels=line_labels, xaxis="Layer", title="Median Cosine Sim of word residuals for the same word", yaxis="Cosine Sim", return_fig=True)
fig.add_hline(1/np.sqrt(d_model), line_dash="dash", line_color="black")
fig.show()
# %%
final_residuals_cent = final_residuals - final_residuals.mean(dim=0)
final_residuals_cent1 = final_residuals_cent[1:, :, ind1, :]
final_residuals_cent1 /= final_residuals_cent1.norm(dim=-1, keepdim=True)
final_residuals_cent2 = final_residuals_cent[:-1, :, ind2, :]
final_residuals_cent2 /= final_residuals_cent2.norm(dim=-1, keepdim=True)
cosine_sim_med = (final_residuals_cent2 * final_residuals_cent1).sum(-1).median(0).values
fig = line(cosine_sim_med.T, line_labels=line_labels, xaxis="Layer", title="Median Cosine Sim of word residuals for different words", yaxis="Cosine Sim", return_fig=True)
fig.add_hline(1/np.sqrt(d_model), line_dash="dash", line_color="black")
fig.show()
# %%
final_residuals_cent_lang = final_residuals[:, :, 1:, :] - final_residuals[:, :, 0:1, :]
final_residuals_cent_lang = final_residuals_cent_lang.mean(0)
final_residuals_cent_lang /= final_residuals_cent_lang.norm(dim=-1, keepdim=True)
ind1 = [0, 0, 1]
ind2 = [1, 2, 2]
x = (final_residuals_cent_lang[:, ind1, :] * final_residuals_cent_lang[:, ind2, :]).sum(-1).T
line(x, line_labels=["Ge-Fr", "Ge-Sp", "Sp-Fr"], title="Cosine Sim of language offset from English", xaxis="Layer", yaxis="Cosine Sim")
# %%
# %%
language = 2
num_words = 8
word_indices = (torch.randperm(len(common_words) - NOUN_INDEX)[:num_words] + NOUN_INDEX).tolist()
lang_prefix = ["EN:", "GE:", "FR:", "SP:"]
eng_prefix = "EN:"
prompt = f"{lang_prefix[language]} "+" ".join([common_words[i][language] for i in word_indices])+"\n"+lang_prefix[0]
answer = " "+" ".join([common_words[i][0] for i in word_indices])
print(prompt)
print(answer)
utils.test_prompt(prompt, answer, model, False)
# %%
text = prompt + answer
tokens = model.to_tokens(text)
logits, cache = model.run_with_cache(tokens)
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice = slice(-num_words, None), return_labels=True)
resid_stack = resid_stack.squeeze(1)
unembed_dirs = model.W_U[:, tokens[0, -num_words:]].T
print(unembed_dirs.shape, resid_stack.shape)
resid_dla = (unembed_dirs * resid_stack).sum(-1).T
resid_dla = torch.cat([resid_dla, resid_dla.mean(0, keepdim=True)])
line(resid_dla, x=resid_labels, line_labels=model.to_str_tokens(tokens[0, -num_words:])+["ave"], title="Residual DLA of French to English translation")
print(resid_dla[:, -n_layers-2:-2].sum(-1))

resid_df = pd.DataFrame(index=resid_labels, data=to_numpy(resid_dla).T, columns=model.to_str_tokens(tokens[0, -num_words:])+["ave"])
resid_df = resid_df.sort_values("ave", ascending=False)
nutils.show_df(resid_df.head(50))
nutils.show_df(resid_df.tail(20))
# %%

token_list = nutils.process_tokens_index(tokens)
top_index = resid_df.head(15).index
heads = []
layers = []
labels = []
for i in top_index:
    if re.match("L(\d+)H(\d+)", i):
        layer, head = re.match("L(\d+)H(\d+)", i).groups()
        layers.append(int(layer))
        heads.append(int(head))
        labels.append(i)

patterns = cache.stack_activation("pattern")[layers, 0, heads, :, :]
imshow(patterns, facet_col=0, facet_labels=labels, x=token_list, y=token_list)
# %%
imshow(patterns[-1], x=token_list, y=token_list)
# %%
# resid_stack, resid_labels = cache.get_full_resid_decomposition(layer=26, expand_neurons=False, apply_ln=True, pos_slice = slice(-num_words, None), return_labels=True)
# resid_stack = resid_stack.squeeze(1)
# unembed_dirs = (model.W_V[26, 11] @ model.W_O[26, 11] @ model.W_U[:, tokens[0, -num_words:]]).T
# print(unembed_dirs.shape, resid_stack.shape)
# resid_dla = (unembed_dirs * resid_stack).sum(-1).T
# resid_dla = torch.cat([resid_dla, resid_dla.mean(0, keepdim=True)])
# line(resid_dla, x=resid_labels, line_labels=model.to_str_tokens(tokens[0, -num_words:])+["ave"], title="Residual DLA of French to English translation")
# print(resid_dla[:, -n_layers-2:-2].sum(-1))

# %%
french_pointer, english_pointer = torch.arange(len(tokens[0]))[tokens[0].cpu() == model.to_single_token(":")].tolist()
french_pointers = []
english_pointers = []
mask = np.zeros((len(tokens[0]), len(tokens[0])))
for i in range(num_words):
    french_word = common_words[word_indices[i]][language]
    num_tokens = final_index[word_indices[i], language]
    french_pointer+=num_tokens
    french_pointers.append(french_pointer)
    english_pointers.append(english_pointer)
    english_pointer+=1
    mask[english_pointers[-1], french_pointers[-1]] = 1
imshow(mask, x=token_list, y=token_list)


# %%
head_df = pd.DataFrame(dict(
    layer = [l for l in range(n_layers) for h in range(n_heads)],
    head = [h for l in range(n_layers) for h in range(n_heads)],
    label = [f"L{l}H{h}" for l in range(n_layers) for h in range(n_heads)],
))
head_df["dla"] = [temp_df.loc[lab]["ave"] for lab in head_df.label]
head_df

# %%
all_patterns = cache.stack_activation("pattern").squeeze(1)
head_df["ind_pattern"] = to_numpy((all_patterns * torch.tensor(mask).cuda()).sum([-1, -2]).flatten()) / num_words
head_df["copy_pattern"] = to_numpy((all_patterns[:, :, 1:] * torch.tensor(mask[:-1]).cuda()).sum([-1, -2]).flatten()) / num_words
nutils.show_df(head_df.sort_values("copy_pattern", ascending=False).head(20))
nutils.show_df(head_df.sort_values("ind_pattern", ascending=False).head(20))
nutils.show_df(head_df.sort_values("dla", ascending=False).head(20))


# %%
top_df = head_df.query("induction>0.25").sort_values("dla", ascending=False).head(5)
layers = top_df.layer.values
heads = top_df["head"].values
labels = top_df.label.values
patterns = all_patterns[layers, heads]
imshow(patterns, facet_col=0, facet_labels=[str(i) for i in labels], x=token_list, y=token_list, title="Patterns of induction-y heads with high DLA")
# %%
def categorise(label):
    if "mlp" in label:
        return "MLP"
    elif label.startswith("L"):
        return "Attn"
    else:
        return label

resid_df["category"] = [categorise(i) for i in resid_df.index]
resid_df.groupby("category").sum()
# %%
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=slice(-num_words-1, -1), 
                                                               return_labels=True)
resid_stack = resid_stack.squeeze(1)
is_mlp = np.array(["mlp" in lab for lab in resid_labels])
is_attn = np.array([lab.startswith("L") for lab in resid_labels])
is_bias = np.array([lab=="bias" for lab in resid_labels])
unembed_dirs = model.W_U[:, tokens[0, -num_words:]]
print(unembed_dirs.shape, resid_stack.shape)
print(np.array(resid_labels)[is_attn])
print(np.array(resid_labels)[is_mlp])
print(np.array(resid_labels)[is_bias])
dlas = resid_stack @ unembed_dirs
attn_dla = dlas[is_attn].sum(0)
mlp_dla = dlas[is_mlp].sum(0)
bias_dla = dlas[is_bias].sum(0)
imshow([attn_dla, mlp_dla, bias_dla], facet_col=0, facet_labels=["Attn", "MLP", "Bias"], x=model.to_str_tokens(tokens[0, -num_words:]), y=model.to_str_tokens(tokens[0, -num_words:]), title="DLA of different components", xaxis="Output", yaxis="Correct next token")

# %%
french_pointer, english_pointer = torch.arange(len(tokens[0]))[tokens[0].cpu() == model.to_single_token(":")].tolist()
french_pointers = []
english_pointers = []
mask = np.zeros((len(tokens[0]), len(tokens[0])))
for i in range(num_words):
    french_word = common_words[word_indices[i]][language]
    num_tokens = final_index[word_indices[i], language]
    french_pointer+=num_tokens
    french_pointers.append(french_pointer)
    english_pointers.append(english_pointer)
    english_pointer+=1

    mask[english_pointers[-1], french_pointers[-1]] = 1
del english_pointer, french_pointer
french_pointers = np.array(french_pointers)
english_pointers = np.array(english_pointers)
print(model.to_str_tokens(tokens[0, english_pointers]))
print(model.to_str_tokens(tokens[0, french_pointers]))
imshow(mask, x=token_list, y=token_list)
# %%
all_values = cache.stack_activation("value")
all_patterns = cache.stack_activation("pattern")
all_values = all_values.squeeze(1)
all_patterns = all_patterns.squeeze(1)
unembed_dirs = model.W_U[:, tokens[0, -num_words:]]
print(all_values.shape, all_patterns.shape)
french_values = all_values[:, french_pointers, :, :]
french_values = einops.rearrange(french_values, "layer french head d_head -> layer head french d_head")
english_values = all_values[:, english_pointers, :, :]
english_values = einops.rearrange(english_values, "layer english head d_head -> layer head english d_head")
ind_french_patterns = all_patterns[:, :, english_pointers, french_pointers]
copy_french_patterns = all_patterns[:, :, english_pointers+1, french_pointers]
self_patterns = all_patterns[:, :, english_pointers, english_pointers]
print(french_values.shape, ind_french_patterns.shape, copy_french_patterns.shape, self_patterns.shape)

W_OU = einops.einsum(model.W_O, unembed_dirs, "layer head d_head d_model, d_model output -> layer head output d_head") / cache["scale"][0, -num_words-1:-1, 0].mean()
french_dla = einops.einsum(french_values, W_OU, "layer head french d_head, layer head output d_head -> layer head french output")
english_dla = einops.einsum(english_values, W_OU, "layer head english d_head, layer head output d_head -> layer head english output")
induction_dla = einops.einsum(french_dla, ind_french_patterns, "layer head french output, layer head french -> layer head french output")
copy_dla = einops.einsum(french_dla, copy_french_patterns, "layer head french output, layer head french -> layer head french output")
self_dla = einops.einsum(english_dla, self_patterns, "layer head english output, layer head english -> layer head english output")
induction_dla_cent = induction_dla - induction_dla.mean(-1, keepdim=True)
copy_dla_cent = copy_dla - copy_dla.mean(-1, keepdim=True)
self_dla_cent = self_dla - self_dla.mean(-1, keepdim=True)
french_dla_cent = french_dla - french_dla.mean(-1, keepdim=True)
english_dla_cent = english_dla - english_dla.mean(-1, keepdim=True)

head_df["induction"] = to_numpy(induction_dla_cent.diagonal(0, -1, -2).mean(-1).flatten())
head_df["induction_uncent"] = to_numpy(induction_dla.diagonal(0, -1, -2).mean(-1).flatten())
head_df["copy"] = to_numpy(copy_dla_cent.diagonal(0, -1, -2).mean(-1).flatten())
head_df["self"] = to_numpy(self_dla_cent.diagonal(0, -1, -2).mean(-1).flatten())
head_df["french"] = to_numpy(french_dla_cent.diagonal(0, -1, -2).mean(-1).flatten())
head_df["english"] = to_numpy(english_dla_cent.diagonal(0, -1, -2).mean(-1).flatten())
nutils.show_df(head_df.sort_values("induction", ascending=False).head(20))
# %%
px.line(head_df, x="label", y=head_df.columns[3:]).show()
px.scatter(head_df, x="induction", y="induction_uncent").show()
px.histogram(head_df, )
# Sanity checking a head
layer = 14
head = 23
OU_temp = W_OU[14, 23]
value = cache["v", layer][0, french_pointers, head, :]
mixed_value = cache["z", layer][0, english_pointers, head, :]
imshow(value @ OU_temp.T)
imshow(mixed_value @ OU_temp.T)
pattern = cache["pattern", layer][0, head]
imshow(pattern, x=token_list, y=token_list)
# %%
temp_value = cache["v", layer][0, :, head, :]
temp_pattern = pattern[english_pointers]
temp_dla = einops.einsum(temp_value, temp_pattern, OU_temp / cache["scale"][0, -num_words-1:-1, 0][:, None], "src_pos d_head, dest_pos src_pos, dest_pos d_head -> dest_pos src_pos")
imshow(temp_dla, x=token_list, y=token_list[-num_words:])
# %%
# Looking at DLA via a head
layer = 14
head = 23
OVU = model.W_V[layer, head] @ model.W_O[layer, head] @ model.W_U

head_french_dlas = final_residuals[:, layer, 2, : ] @ OVU
for i in range(0, len(common_words), 7):
    print(common_words[i])
    temp_df = nutils.create_vocab_df(head_french_dlas[i])
    temp_df["rank"] = np.arange(len(temp_df))
    print(temp_df.loc[model.to_single_token(" "+common_words[i][0])])
    nutils.show_df(temp_df.head(10))
# %%
english_tokens = [model.to_single_token(" "+common_words[i][0]) for i in range(len(common_words))]
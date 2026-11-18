"""
Model definitions used in the MetaNCF paper (Amazon Fashion experiments).

The classes are copied unchanged from the original research code (h_model.py) so that the released
models are identical to those used for the reported results:
  NeuMF          - NCF (GMF + MLP), original NeuMF sizes (GMF dim 8, MLP dim 32, MLP 64-32-16-8)
  NeuMF_ablation - NCF with only the GMF or only the MLP path (use exactly one of gmf_use / mlp_use)
  NeuMF_Concat   - NCF-Concat: item attributes concatenated to the MLP input (static fusion)
  MetaNCF        - MetaNCF with raw (float) numeric attributes; the paper uses fusion_type='concat'
  MetaNCF_bin    - MetaNCF with binned numeric attributes (embedding look-ups) and attention temperature tau
All models take the same six inputs: [user_idx, item_idx, store_idx, average_rating, rating_number, price].
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers


class NeuMF(Model):
    def __init__(self, num_users, num_items, num_stores=None, fusion_type=None,
                 layers_mlp=(64, 32, 16, 8), reg_mf=1e-2, reg_mlp=1e-2):
        super(NeuMF, self).__init__()

        # 1. GMF Part with L2 Regularization
        self.embed_user_GMF = layers.Embedding(
            num_users, layers_mlp[-1], name='user_gmf',
            embeddings_initializer='uniform',
            embeddings_regularizer=regularizers.l2(reg_mf)
        )
        self.embed_item_GMF = layers.Embedding(
            num_items, layers_mlp[-1], name='item_gmf',
            embeddings_initializer='uniform',
            embeddings_regularizer=regularizers.l2(reg_mf)
        )

        # 2. MLP Part with L2 Regularization
        mlp_dim = layers_mlp[0] // 2
        self.embed_user_MLP = layers.Embedding(
            num_users, mlp_dim, name='user_mlp',
            embeddings_initializer='uniform',
            embeddings_regularizer=regularizers.l2(reg_mlp)
        )
        self.embed_item_MLP = layers.Embedding(
            num_items, mlp_dim, name='item_mlp',
            embeddings_initializer='uniform',
            embeddings_regularizer=regularizers.l2(reg_mlp)
        )

        self.mlp_layers = []
        for units in layers_mlp:
            self.mlp_layers.append(layers.Dense(
                units, activation='relu',
                kernel_regularizer=regularizers.l2(reg_mlp)
            ))

        # 3. Final Prediction Layer
        self.prediction_layer = layers.Dense(1, activation=None, kernel_initializer='lecun_uniform')

    def call(self, inputs):
        user_indices, item_indices, _, _, _, _ = inputs

        # GMF Path
        user_gmf = self.embed_user_GMF(user_indices)
        item_gmf = self.embed_item_GMF(item_indices)
        mf_output = layers.Multiply()([user_gmf, item_gmf])

        # MLP Path
        user_mlp = self.embed_user_MLP(user_indices)
        item_mlp = self.embed_item_MLP(item_indices)
        mlp_output = layers.Concatenate()([user_mlp, item_mlp])
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)

        # Combine
        combined = layers.Concatenate()([mf_output, mlp_output])
        prediction = self.prediction_layer(combined)

        return prediction


class NeuMF_ablation(Model):
    def __init__(self, num_users, num_items, num_stores=None, fusion_type=None,
                 layers_mlp=(64, 32, 16, 8), reg_mf=1e-2, reg_mlp=1e-2, gmf_use=True, mlp_use=True):
        super(NeuMF_ablation, self).__init__()

        self.gmf_use = gmf_use
        self.mlp_use = mlp_use

        # 1. GMF Part with L2 Regularization
        self.embed_user_GMF = layers.Embedding(
            num_users, layers_mlp[-1], name='user_gmf',
            embeddings_initializer='uniform',
            embeddings_regularizer=regularizers.l2(reg_mf)
        )
        self.embed_item_GMF = layers.Embedding(
            num_items, layers_mlp[-1], name='item_gmf',
            embeddings_initializer='uniform',
            embeddings_regularizer=regularizers.l2(reg_mf)
        )

        # 2. MLP Part with L2 Regularization
        mlp_dim = layers_mlp[0] // 2
        self.embed_user_MLP = layers.Embedding(
            num_users, mlp_dim, name='user_mlp',
            embeddings_initializer='uniform',
            embeddings_regularizer=regularizers.l2(reg_mlp)
        )
        self.embed_item_MLP = layers.Embedding(
            num_items, mlp_dim, name='item_mlp',
            embeddings_initializer='uniform',
            embeddings_regularizer=regularizers.l2(reg_mlp)
        )

        self.mlp_layers = []
        for units in layers_mlp:
            self.mlp_layers.append(layers.Dense(
                units, activation='relu',
                kernel_regularizer=regularizers.l2(reg_mlp)
            ))

        # 3. Final Prediction Layer
        self.prediction_layer = layers.Dense(1, activation=None, kernel_initializer='lecun_uniform')

    def call(self, inputs):
        user_indices, item_indices, _, _, _, _ = inputs

        # GMF Path
        user_gmf = self.embed_user_GMF(user_indices)
        item_gmf = self.embed_item_GMF(item_indices)
        mf_output = layers.Multiply()([user_gmf, item_gmf])

        # MLP Path
        user_mlp = self.embed_user_MLP(user_indices)
        item_mlp = self.embed_item_MLP(item_indices)
        mlp_output = layers.Concatenate()([user_mlp, item_mlp])
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)

        # Combine
        if self.mlp_use and self.gmf_use:
            combined = layers.Concatenate()([mf_output, mlp_output])
            prediction = self.prediction_layer(combined)

        if self.mlp_use:
            prediction = self.prediction_layer(mlp_output)

        if self.gmf_use:
            prediction = self.prediction_layer(mf_output)

        return prediction



#import tensorflow as tf
#from tensorflow.keras import layers, Model, regularizers


class NeuMF_Concat(Model):
    def __init__(self, num_users, num_items, num_stores, fusion_type=None, latent_dim=16,
                 layers_mlp=(64, 32, 32, 16), reg_mf=1e-2, reg_mlp=1e-2):
        super(NeuMF_Concat, self).__init__()

        # 1. Collaborative Embeddings
        self.embed_user_GMF = layers.Embedding(num_users, latent_dim,
                                               embeddings_regularizer=regularizers.l2(reg_mf))
        self.embed_item_GMF = layers.Embedding(num_items, latent_dim,
                                               embeddings_regularizer=regularizers.l2(reg_mf))

        self.embed_user_MLP = layers.Embedding(num_users, latent_dim,
                                               embeddings_regularizer=regularizers.l2(reg_mlp))
        self.embed_item_MLP = layers.Embedding(num_items, latent_dim,
                                               embeddings_regularizer=regularizers.l2(reg_mlp))

        # 2. Meta-Attribute Embeddings
        self.embed_store = layers.Embedding(num_stores, 8,
                                            embeddings_regularizer=regularizers.l2(reg_mlp))

        # 3. MLP Backbone
        self.mlp_layers = []
        for units in layers_mlp:
            self.mlp_layers.append(layers.Dense(units, activation='relu',
                                                kernel_regularizer=regularizers.l2(reg_mlp)))

        # 4. Final Prediction Layer
        self.prediction_layer = layers.Dense(1, activation=None, kernel_initializer='lecun_uniform')

    def call(self, inputs):
        """
        Input order: [user_idx, item_idx, store_idx, avg_rating, rating_num, price]
        """
        u_idx, i_idx, s_idx, avg_r, r_num, price = inputs

        # Reshape scalar inputs to (Batch, 1) to ensure they can be concatenated
        avg_r = tf.expand_dims(avg_r, -1)
        r_num = tf.expand_dims(r_num, -1)
        price = tf.expand_dims(price, -1)

        # --- GMF Path ---
        u_gmf = self.embed_user_GMF(u_idx)
        i_gmf = self.embed_item_GMF(i_idx)
        mf_output = layers.Multiply()([u_gmf, i_gmf])

        # --- MLP Path ---
        u_mlp = self.embed_user_MLP(u_idx)
        i_mlp = self.embed_item_MLP(i_idx)
        s_emb = self.embed_store(s_idx)

        # Internal Concatenation of all Item Meta-Attributes
        item_combined = layers.Concatenate()([i_mlp, s_emb, avg_r, r_num, price])

        mlp_input = layers.Concatenate()([u_mlp, item_combined])

        x = mlp_input
        for layer in self.mlp_layers:
            x = layer(x)

        # --- Final Merge ---
        combined = layers.Concatenate()([mf_output, x])
        return self.prediction_layer(combined)


class MetaNCF(Model):
    def __init__(self, num_users, num_items, num_stores, latent_dim=16, num_heads=4,
                 layers_mlp=[64, 32, 32, 16], reg_mf=1e-2, reg_mlp=1e-2,
                 fusion_type='residual'):
        """
        MetaNCF with four separate meta-attribute attention slots.
        """
        super(MetaNCF, self).__init__()
        self.fusion_type = fusion_type.lower()
        self.latent_dim = latent_dim

        # 1. Base Latent Embeddings (Collaborative Signal)
        self.user_emb_gmf = layers.Embedding(num_users, latent_dim, name='u_gmf',
                                             embeddings_regularizer=regularizers.l2(reg_mf))
        self.item_emb_gmf = layers.Embedding(num_items, latent_dim, name='i_gmf',
                                             embeddings_regularizer=regularizers.l2(reg_mf))

        self.user_emb_mlp = layers.Embedding(num_users, latent_dim, name='u_mlp',
                                             embeddings_regularizer=regularizers.l2(reg_mlp))
        self.item_emb_mlp = layers.Embedding(num_items, latent_dim, name='i_mlp',
                                             embeddings_regularizer=regularizers.l2(reg_mlp))

        # 2. Meta-Attribute Projections (Content Signal)
        self.store_emb = layers.Embedding(num_stores, latent_dim, name='store_emb',
                                          embeddings_regularizer=regularizers.l2(reg_mlp))

        # We use separate projection layers for each numerical attribute
        self.price_proj = layers.Dense(latent_dim, activation='relu', name='price_proj')
        self.avg_rating_proj = layers.Dense(latent_dim, activation='relu', name='avg_rating_proj')
        self.rating_num_proj = layers.Dense(latent_dim, activation='relu', name='rating_num_proj')

        # 3. Multi-Head Attention Layer
        # key_dim is the size of each head. usually latent_dim // num_heads
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=latent_dim // num_heads,
            kernel_regularizer=regularizers.l2(reg_mlp),
            name='mha_unit'
        )

        # Layer Normalization is recommended after MHA for stability
        self.layernorm = layers.LayerNormalization()

        # 4. Fusion Layer Components
        if self.fusion_type == 'gated':
            self.fusion_gate = layers.Dense(1, activation='sigmoid', name='gate')
        elif self.fusion_type == 'residual':
            self.res_proj = layers.Dense(latent_dim, name='res_projection')
        elif self.fusion_type == 'concat':
            self.concat_proj = layers.Dense(latent_dim, name='concat_projection')

        # 5. MLP Backbone
        self.mlp_layers = []
        for i, units in enumerate(layers_mlp):
            self.mlp_layers.append(layers.Dense(units, activation='relu', name=f'mlp_layer_{i}',
                                                kernel_regularizer=regularizers.l2(reg_mlp)))

        # 6. Final Prediction (Regression output)
        self.prediction_layer = layers.Dense(1, activation=None, kernel_initializer='lecun_uniform', name='prediction')

    def call(self, inputs):
        # Unpack inputs from final_df
        u_idx, i_idx, s_idx, avg_r, r_num, price = inputs

        # --- STEP 1: Collaborative Embedding Lookups ---
        u_gmf = self.user_emb_gmf(u_idx)
        i_gmf = self.item_emb_gmf(i_idx)
        u_mlp = self.user_emb_mlp(u_idx)
        i_mlp = self.item_emb_mlp(i_idx)

        # --- STEP 2: Meta-Attribute Processing ---
        s_vec = self.store_emb(s_idx)
        # Each numerical attribute is expanded and projected independently
        p_vec = self.price_proj(tf.expand_dims(price, -1))
        ar_vec = self.avg_rating_proj(tf.expand_dims(avg_r, -1))
        rn_vec = self.rating_num_proj(tf.expand_dims(r_num, -1))

        # --- STEP 3: Attention (4 separate meta-slots) ---
        # Attributes to attend to: [Batch, 4, latent_dim]
        meta_stack = tf.stack([s_vec, p_vec, ar_vec, rn_vec], axis=1)

        # Multi-Head Attention
        # Q: User [Batch, 1, latent_dim], K/V: Meta [Batch, 4, latent_dim]
        query = tf.expand_dims(u_mlp, axis=1)

        # MHA output shape: [Batch, 1, latent_dim]
        attn_output = self.mha(query=query, value=meta_stack, key=meta_stack)

        # Residual connection + LayerNorm (Standard Transformer Block)
        attentive_meta = tf.squeeze(attn_output, axis=1)
        attentive_meta = self.layernorm(attentive_meta + u_mlp)

        # --- STEP 4: Information Fusion ---
        if self.fusion_type == 'additive':
            i_final_gmf = i_gmf + attentive_meta
            i_final_mlp = i_mlp + attentive_meta
        elif self.fusion_type == 'residual':
            refined_meta = self.res_proj(attentive_meta)
            i_final_gmf = i_gmf + refined_meta
            i_final_mlp = i_mlp + refined_meta
        elif self.fusion_type == 'gated':
            g = self.fusion_gate(tf.concat([i_mlp, attentive_meta], axis=-1))
            i_final_gmf = g * i_gmf + (1 - g) * attentive_meta
            i_final_mlp = g * i_mlp + (1 - g) * attentive_meta
        elif self.fusion_type == 'concat':
            # Simple projection to maintain dimension symmetry
            i_final_gmf = self.concat_proj(tf.concat([i_gmf, attentive_meta], axis=-1))
            i_final_mlp = self.concat_proj(tf.concat([i_mlp, attentive_meta], axis=-1))

        # --- STEP 5: Dual-Path Interaction ---
        gmf_out = tf.multiply(u_gmf, i_final_gmf)
        mlp_in = tf.concat([u_mlp, i_final_mlp], axis=-1)

        x = mlp_in
        for layer in self.mlp_layers:
            x = layer(x)
        mlp_out = x

        # --- STEP 6: Final Prediction ---
        combined = tf.concat([gmf_out, mlp_out], axis=-1)
        return self.prediction_layer(combined)


class MetaNCF_bin(Model):
    def __init__(self, num_users, num_items, num_stores, num_bins, latent_dim=16, num_heads=4,
                 layers_mlp=[64, 32, 32, 16], reg_mf=1e-2, reg_mlp=1e-2, tau=0.01,
                 fusion_type='residual'):
        """
        MetaNCF with four separate meta-attribute attention slots.
        """
        super(MetaNCF_bin, self).__init__()
        self.fusion_type = fusion_type.lower()
        self.latent_dim = latent_dim
        self.tau = tau

        # 1. Base Latent Embeddings (Collaborative Signal)
        self.user_emb_gmf = layers.Embedding(num_users, latent_dim, name='u_gmf',
                                             embeddings_regularizer=regularizers.l2(reg_mf))
        self.item_emb_gmf = layers.Embedding(num_items, latent_dim, name='i_gmf',
                                             embeddings_regularizer=regularizers.l2(reg_mf))

        self.user_emb_mlp = layers.Embedding(num_users, latent_dim, name='u_mlp',
                                             embeddings_regularizer=regularizers.l2(reg_mlp))
        self.item_emb_mlp = layers.Embedding(num_items, latent_dim, name='i_mlp',
                                             embeddings_regularizer=regularizers.l2(reg_mlp))

        # 2. Meta-Attribute Projections (Content Signal)
        self.store_emb = layers.Embedding(num_stores, latent_dim, name='store_emb',
                                          embeddings_regularizer=regularizers.l2(reg_mlp))
        self.price_emb = layers.Embedding(num_bins, latent_dim, name='price_emb',
                                          embeddings_regularizer=regularizers.l2(reg_mlp))
        self.average_rating_emb = layers.Embedding(num_bins, latent_dim, name='avg_rating_emb',
                                          embeddings_regularizer=regularizers.l2(reg_mlp))
        self.rating_num_emb = layers.Embedding(num_bins, latent_dim, name='rating_num_emb',
                                          embeddings_regularizer=regularizers.l2(reg_mlp))


        # 3. Multi-Head Attention Layer
        # key_dim is the size of each head. usually latent_dim // num_heads
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=latent_dim // num_heads,
            kernel_regularizer=regularizers.l2(reg_mlp),
            name='mha_unit'
        )

        # Layer Normalization is recommended after MHA for stability
        self.layernorm = layers.LayerNormalization()

        # 4. Fusion Layer Components
        if self.fusion_type == 'gated':
            self.fusion_gate = layers.Dense(1, activation='sigmoid', name='gate')
        elif self.fusion_type == 'residual':
            self.res_proj = layers.Dense(latent_dim, name='res_projection')
        elif self.fusion_type == 'concat':
            self.concat_proj = layers.Dense(latent_dim, name='concat_projection')

        # 5. MLP Backbone
        self.mlp_layers = []
        for i, units in enumerate(layers_mlp):
            self.mlp_layers.append(layers.Dense(units, activation='relu', name=f'mlp_layer_{i}',
                                                kernel_regularizer=regularizers.l2(reg_mlp)))

        # 6. Final Prediction (Regression output)
        self.prediction_layer = layers.Dense(1, activation=None, kernel_initializer='lecun_uniform', name='prediction')

        self.shared_space = layers.Dense(latent_dim, activation='tanh')


    def call(self, inputs):
        # Unpack inputs from final_df
        u_idx, i_idx, s_idx, avg_r, r_num, price = inputs

        # --- STEP 1: Collaborative Embedding Lookups ---
        u_gmf = self.user_emb_mlp(u_idx)  # previously separate embedding emb_gmf,
        i_gmf = self.item_emb_gmf(i_idx)
        u_mlp = self.user_emb_mlp(u_idx)
        i_mlp = self.item_emb_mlp(i_idx)

        # --- STEP 2: Meta-Attribute Processing ---
        s_vec = self.store_emb(s_idx)
        # Each numerical attribute is expanded and projected independently
        p_vec = self.price_emb(price)
        ar_vec = self.average_rating_emb(avg_r)
        rn_vec = self.rating_num_emb(r_num)

        # --- STEP 3: Attention (4 separate meta-slots) ---
        # Attributes to attend to: [Batch, 4, latent_dim]
        meta_stack = tf.stack([s_vec, p_vec, ar_vec, rn_vec], axis=1)

        # Multi-Head Attention
        # Q: User [Batch, 1, latent_dim], K/V: Meta [Batch, 4, latent_dim]
        query = tf.expand_dims(u_mlp, axis=1)

        query = self.shared_space(query)
        meta_stack = self.shared_space(meta_stack)

        # MHA output shape: [Batch, 1, latent_dim]
        attn_output = self.mha(query=query/self.tau, value=meta_stack, key=meta_stack)

        # Residual connection + LayerNorm (Standard Transformer Block)
        attentive_meta = tf.squeeze(attn_output, axis=1)
        attentive_meta = self.layernorm(attentive_meta + u_mlp)

        # --- STEP 4: Information Fusion ---
        if self.fusion_type == 'additive':
            i_final_gmf = i_gmf + attentive_meta
            i_final_mlp = i_mlp + attentive_meta
        elif self.fusion_type == 'residual':
            refined_meta = self.res_proj(attentive_meta)
            i_final_gmf = i_gmf + refined_meta
            i_final_mlp = i_mlp + refined_meta
        elif self.fusion_type == 'gated':
            g = self.fusion_gate(tf.concat([i_mlp, attentive_meta], axis=-1))
            i_final_gmf = g * i_gmf + (1 - g) * attentive_meta
            i_final_mlp = g * i_mlp + (1 - g) * attentive_meta
        elif self.fusion_type == 'concat':
            # Simple projection to maintain dimension symmetry
            i_final_gmf = self.concat_proj(tf.concat([i_gmf, attentive_meta], axis=-1))
            i_final_mlp = self.concat_proj(tf.concat([i_mlp, attentive_meta], axis=-1))

        # --- STEP 5: Dual-Path Interaction ---
        gmf_out = tf.multiply(u_gmf, i_final_gmf)
        mlp_in = tf.concat([u_mlp, i_final_mlp], axis=-1)

        x = mlp_in
        for layer in self.mlp_layers:
            x = layer(x)
        mlp_out = x

        # --- STEP 6: Final Prediction ---
        combined = tf.concat([gmf_out, mlp_out], axis=-1)
        return self.prediction_layer(combined)


def build_model(name, num_users, num_items, num_stores, l2, fusion='concat', latent_dim=16, num_heads=4,
                tau=0.01, num_bins=10):
    """Factory for the models of the ablation (Table 6) and the binned variant."""
    if name == 'ncf':
        return NeuMF(num_users, num_items, reg_mlp=l2, reg_mf=l2)
    if name == 'ncf_mlp':
        return NeuMF_ablation(num_users, num_items, reg_mlp=l2, reg_mf=l2, gmf_use=False, mlp_use=True)
    if name == 'ncf_gmf':
        return NeuMF_ablation(num_users, num_items, reg_mlp=l2, reg_mf=l2, gmf_use=True, mlp_use=False)
    if name == 'ncf_concat':
        return NeuMF_Concat(num_users, num_items, num_stores, reg_mlp=l2, reg_mf=l2)
    if name == 'metancf':
        return MetaNCF(num_users, num_items, num_stores, latent_dim=latent_dim, num_heads=num_heads,
                       fusion_type=fusion, reg_mlp=l2, reg_mf=l2)
    if name == 'metancf_bin':
        return MetaNCF_bin(num_users, num_items, num_stores, num_bins=num_bins, latent_dim=latent_dim,
                           num_heads=num_heads, tau=tau, fusion_type=fusion, reg_mlp=l2, reg_mf=l2)
    raise ValueError(f'unknown model: {name}')


MODEL_NAMES = ['ncf_mlp', 'ncf_gmf', 'ncf', 'ncf_concat', 'metancf', 'metancf_bin']
BINNED_MODELS = {'metancf_bin'}

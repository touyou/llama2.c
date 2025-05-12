// Original: https://gist.github.com/kishida/05656bfcbe840f269784f7dbbee5928e
// based on https://github.com/karpathy/llama2.c/commit/411c5bd2db9a87e94e1bd1a6c7b7ca117adc4b01
// at Sep 14, 2023

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.io.UnsupportedEncodingException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.IntStream;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;

public class Llama {
// ----------------------------------------------------------------------------
// Transformer model

static class Config{
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length

    /*
    static StructLayout layout = MemoryLayout.structLayout(
        JAVA_INT.withName("dim"),
        JAVA_INT.withName("hidden_dim"),
        JAVA_INT.withName("n_layers"),
        JAVA_INT.withName("n_heads"),
        JAVA_INT.withName("n_kv_heads"),
        JAVA_INT.withName("vocab_size"),
        JAVA_INT.withName("seq_len")
    );
    */

    void load(SegmentAllocator alloc) {
        dim = alloc.allocate(JAVA_INT).get(JAVA_INT, 0);
        hidden_dim = alloc.allocate(JAVA_INT).get(JAVA_INT, 0);
        n_layers = alloc.allocate(JAVA_INT).get(JAVA_INT, 0);
        n_heads = alloc.allocate(JAVA_INT).get(JAVA_INT, 0);
        n_kv_heads = alloc.allocate(JAVA_INT).get(JAVA_INT, 0);
        vocab_size = alloc.allocate(JAVA_INT).get(JAVA_INT, 0);
        seq_len = alloc.allocate(JAVA_INT).get(JAVA_INT, 0);
    }

    @Override
    public String toString() {
        return "dim:%d hidden_dim:%d n_layers:%d n_heads:%d n_kv_heads:%d vocab_size:%d seq_len:%d"
            .formatted(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len);
    }

}
static FloatBuffer toBuffer(MemorySegment seg) {
    return seg.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
}
static class TransformerWeights{
    // token embedding table
    MemorySegment token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
    FloatBuffer[] rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    FloatBuffer[] wq; // (layer, dim, n_heads * head_size)
    FloatBuffer[] wk; // (layer, dim, n_kv_heads * head_size)
    FloatBuffer[] wv; // (layer, dim, n_kv_heads * head_size)
    FloatBuffer[] wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    FloatBuffer[] w1; // (layer, hidden_dim, dim)
    FloatBuffer[] w2; // (layer, dim, hidden_dim)
    FloatBuffer[] w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    FloatBuffer rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    MemorySegment wcls;



    static FloatBuffer[] alloc(SegmentAllocator allocator, int layers, int size) {
        return  IntStream.range(0, layers)
                .mapToObj(i -> allocator.allocate(JAVA_FLOAT, size))
                .map(Llama::toBuffer)
                .toArray(FloatBuffer[]::new);
    }

    void memory_map_weights(Config p, SegmentAllocator allocator, boolean shared_weights) {
        int head_size = p.dim / p.n_heads;
        // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
        int n_layers = p.n_layers;
        this.token_embedding_table = allocator.allocate(JAVA_FLOAT, p.vocab_size * p.dim);
        this.rms_att_weight = alloc(allocator, n_layers, p.dim);
        this.wq = alloc(allocator, n_layers, p.dim * p.dim);
        this.wk = alloc(allocator, n_layers, p.dim * (p.n_kv_heads * head_size));
        this.wv = alloc(allocator, n_layers, p.dim * (p.n_kv_heads * head_size));
        this.wo = alloc(allocator, n_layers, p.dim * p.dim);
        this.rms_ffn_weight = alloc(allocator, n_layers, p.dim);
        this.w1 = alloc(allocator, n_layers, p.dim * p.hidden_dim);
        this.w2 = alloc(allocator, n_layers, p.hidden_dim * p.dim);
        this.w3 = alloc(allocator, n_layers, p.dim * p.hidden_dim);
        this.rms_final_weight = toBuffer(allocator.allocate(JAVA_FLOAT, p.dim));
        allocator.allocate(JAVA_FLOAT, p.seq_len * head_size / 2);// skip what used to be freq_cis_real (for RoPE)
        allocator.allocate(JAVA_FLOAT, p.seq_len * head_size / 2);// skip what used to be freq_cis_imag (for RoPE)
        this.wcls = shared_weights ? this.token_embedding_table : allocator.allocate(JAVA_FLOAT, p.vocab_size * p.dim);
    }

}

static class RunState{
    // current wave of activations
    //float[] x; // activation at current time stamp (dim,)
    float[] xb; // same, but inside a residual branch (dim,)
    float[] xb2; // an additional buffer just for convenience (dim,)
    float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float[] q; // query (dim,)
    float[] k; // key (dim,)
    float[] v; // value (dim,)
    float[][] att; // buffer for scores/attention values (n_heads, seq_len)
    float[] logits; // output logits
    // kv cache
    float[][][] key_cache;   // (layer, seq_len, dim)
    float[][][] value_cache; // (layer, seq_len, dim)


    private float[] calloc(int size) {
        //return offHeap.allocate(ValueLayout.JAVA_FLOAT, size);
        return new float[size];
    }

    void malloc(Config p) {

        // we calloc instead of malloc to keep valgrind happy
        int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
        //this.x = calloc(p.dim);
        this.xb = calloc(p.dim);
        this.xb2 = calloc(p.dim);
        this.hb = calloc(p.hidden_dim);
        this.hb2 = calloc(p.hidden_dim);
        this.q = calloc(p.dim);
        this.key_cache = new float[p.n_layers][p.seq_len][kv_dim];
        this.value_cache = new float[p.n_layers][p.seq_len][kv_dim];
        this.att = new float[p.n_heads][p.seq_len];
        this.logits = calloc(p.vocab_size);
    }

    void free() {
    }
}

static class CustomSlicingAllocator implements SegmentAllocator {
    private final MemorySegment segment;
    private long sp = 0L;
    public CustomSlicingAllocator(MemorySegment segment) {
        this.segment = segment;
    }
    MemorySegment trySlice(long byteSize, long byteAlignment) {
        MemorySegment slice = segment.asSlice(sp, byteSize, byteAlignment);
        sp += byteSize;
        return slice;
    }
    @Override
    public MemorySegment allocate(long byteSize, long byteAlignment) {
        return trySlice(byteSize, byteAlignment);
    }
}
static boolean vectorize = false;

static class Transformer {
    Config config = new Config(); // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights = new TransformerWeights(); // the weights of the model
    RunState state = new RunState(); // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    FileChannel fd; // file descriptor for memory mapping
    MemorySegment data; // memory mapped data pointer

    Arena arena;

    void read_checkpoint(String checkpoint) throws IOException {
        var path = Path.of(checkpoint);
        fd = FileChannel.open(path, Set.of(StandardOpenOption.READ));
        arena = Arena.ofShared();
        data = fd.map(FileChannel.MapMode.READ_ONLY, 0, Files.size(path), arena);

        SegmentAllocator alloc = new CustomSlicingAllocator(data);
        config.load(alloc);
        boolean shared_weights = config.vocab_size > 0;
        config.vocab_size = Math.abs(config.vocab_size);
        weights.memory_map_weights(config, alloc, shared_weights);
    }

    void build(String checkpointPath) {
        try {
            read_checkpoint(checkpointPath);
            state.malloc(config);
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }
    void free() {
        try {
            arena.close();
            fd.close();
            state.free();
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

static void rmsnorm(float[] out, float[] x, FloatBuffer weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        float f = x[j];
        ss += f * f;
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / (float)Math.sqrt(ss); //
    // normalize and scale
    for (int j = 0; j < size; j++) {
        out[j] = weight.get(j) * (ss * x[j]);
    }
}

static void softmax(float[] x, int size) {
    final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        max_val = Float.max(x[i], max_val);
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i] = (float)Math.exp(x[i] - max_val);
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
static final int SIMD_SIZE = SPECIES.length();
static final long FLOAT_SIZE = ValueLayout.JAVA_FLOAT.byteSize();
static void matmul(float[] xout, float[] x, FloatBuffer ws, int n, int d) {
    // W (d,n) @ x (n,) . xout (d,)
    // by far the most amount of time is spent inside this little function

    MemorySegment w = MemorySegment.ofBuffer(ws);
    IntStream.range(0, d).parallel().forEach(i -> {
        if (vectorize) {
            FloatVector val = FloatVector.zero(SPECIES);

            for (int j = 0; j < n; j+=SIMD_SIZE) {
                // val += get(w, i * n + j) * get(x, j);
                FloatVector a = FloatVector.fromMemorySegment(SPECIES, w, (i * n + j + 0*SIMD_SIZE) * FLOAT_SIZE, ByteOrder.LITTLE_ENDIAN);
                FloatVector b = FloatVector.fromArray(SPECIES, x, j + 0*SIMD_SIZE);
                val = a.fma(b, val);
            }
            xout[i] = val.reduceLanes(VectorOperators.ADD);
        } else {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += ws.get(i * n + j) * x[j];
            }
            xout[i] = val;
        }

    });
}

static MemorySegment slice(MemorySegment data, int index, int size) {
    long sliceSize = size * JAVA_FLOAT.byteSize();
    return data.asSlice(index * sliceSize, sliceSize);
}

static float[] forward(Transformer transformer, int token, int pos) {

    // a few convenience variables
    Config p = transformer.config;
    TransformerWeights w = transformer.weights;
    RunState s = transformer.state;

    int dim = p.dim;
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    int kv_mul = p.n_heads / p.n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p.hidden_dim;
    int head_size = dim / p.n_heads;
    var head_aqrt = (float)Math.sqrt(head_size);
    // copy the token embedding into x
    MemorySegment content_row = slice(w.token_embedding_table, token, dim);
    float[] x = content_row.toArray(JAVA_FLOAT);

    // forward all the layers
    for(int l = 0; l < p.n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight[l], dim);

        // key and value point to the kv cache
        s.k = s.key_cache[l][pos];
        s.v = s.value_cache[l][pos];

        // qkv matmuls for this position
        matmul(s.q, s.xb, w.wq[l], dim, dim);
        matmul(s.k, s.xb, w.wk[l], dim, kv_dim);
        matmul(s.v, s.xb, w.wv[l], dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / (float)Math.pow(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = (float)Math.cos(val);
            float fci = (float)Math.sin(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float[] vec = v == 0 ? s.q : s.k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        final var fl = l;
        IntStream.range(0, p.n_heads).parallel().forEach(h -> {

            // get the query vector for this head
            int qpos = h * head_size;
            int kvpos = h / kv_mul * head_size;
            // attention scores for this head
            float[] att = s.att[h];
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // calculate the attention score as the dot product of q and k
                float score = 0;
                if (true && vectorize) {
                    FloatVector val = FloatVector.zero(SPECIES);
                    for (int i = 0; i < head_size; i+=SIMD_SIZE) {
                        FloatVector a = FloatVector.fromArray(SPECIES, s.q, qpos + i);
                        FloatVector b = FloatVector.fromArray(SPECIES, s.key_cache[fl][t], kvpos + i);
                        val = a.fma(b, val);
                    }
                    score = val.reduceLanes(VectorOperators.ADD);
                } else {
                    for (int i = 0; i < head_size; i++) {
                        score += s.q[qpos + i] * s.key_cache[fl][t][kvpos + i];
                    }
                }
                score /= head_aqrt;
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            Arrays.fill(s.xb, h * head_size, (h + 1) * head_size, 0);
            for (int t = 0; t <= pos; t++) {
                if (false && vectorize) { // this vectorize makes generation slow
                    FloatVector aVec = FloatVector.broadcast(SPECIES, att[t]);

                    for (int i = 0; i < head_size; i += SIMD_SIZE) {
                        FloatVector xbVec = FloatVector.fromArray(SPECIES, s.xb, qpos + i);
                        FloatVector valVec = FloatVector.fromArray(SPECIES, s.value_cache[fl][t], kvpos + i);
                        // FMA: (a * valVec) + xbVec
                        FloatVector result = aVec.fma(valVec, xbVec);
                        result.intoArray(s.xb, qpos + i);
                    }
                } else {
                    // get the attention weight for this timestep
                    float a = att[t];
                    // accumulate the weighted value into xb
                    for (int i = 0; i < head_size; i++) {
                        s.xb[qpos + i] += a * s.value_cache[fl][t][kvpos + i];
                    }
                }
            }
        });

        // final matmul to get the output of the attention
        matmul(s.xb2, s.xb, w.wo[l], dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s.xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight[l], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s.hb, s.xb, w.w1[l], dim, hidden_dim);
        matmul(s.hb2, s.xb, w.w3[l], dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s.hb[i];
            // silu(x)=x*ﾏ・x), where ﾏ・x) is the logistic sigmoid
            val *= (1.0f / (1.0f + Math.exp(-val)));
            // elementwise multiply with w3(x)
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s.xb, s.hb, w.w2[l], hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s.xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w.rms_final_weight, dim);

    // classifier into logits
    matmul(s.logits, x, toBuffer(w.wcls), p.dim, p.vocab_size);
    return s.logits;
}


// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <. tokens

static class TokenIndex implements Comparable<TokenIndex>{
    byte[] str;
    int id;

    @Override
    public int compareTo(TokenIndex o) {
        for (int i = 0; i < Math.min(str.length, o.str.length); ++i) {
            if (str[i] != o.str[i]) {
                return str[i] < o.str[i] ? -1 : 1;
            }
            if (str[i] == 0) {
                return 0;
            }
        }
        return Integer.compare(str.length, o.str.length);
    }
}

static class Tokenizer {
    byte[][] vocab;
    byte[][] slided; // slided cache
    float[] vocab_scores;
    TokenIndex[] sorted_vocab;
    int vocab_size;
    int max_token_length;
    byte[][] byte_pieces = new byte[256][]; // stores all single-byte strings

    static int readInt(byte[] buf) {
        return buf[0] | buf[1] << 8 | buf[2] << 16 | buf[3] << 24;
    }

    void build(String tokenizer_path, int vocab_size) {
        // i should have written the vocab_size into the tokenizer file... sigh
        this.vocab_size = vocab_size;
        // malloc space to hold the scores and the strings
        this.vocab = new byte[vocab_size][];
        this.slided = new byte[vocab_size][];
        this.vocab_scores = new float[vocab_size];
        this.sorted_vocab = null; // initialized lazily
        for (int i = 0; i < 256; i++) {
            this.byte_pieces[i] = new byte[]{(byte)i, (byte)0};
        }
        // read in the file
        byte[] buf = new byte[4];
        try (InputStream file = Files.newInputStream(Path.of(tokenizer_path))) {
            file.read(buf);
            this.max_token_length = readInt(buf);
            //System.out.println(this.max_token_length);
            for (int i = 0; i < vocab_size; i++) {
                file.read(buf);
                this.vocab_scores[i] = Float.intBitsToFloat(readInt(buf));
                file.read(buf);
                int len = readInt(buf);
                //System.out.println(len);
                this.vocab[i] = new byte[len + 1];
                file.read(this.vocab[i], 0, len);
                this.vocab[i][len] = '\0'; // add the string terminating token
            }

        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
    }

    void free() {
        // do nothing
    }
}

static int hex2int(int c) {
    if (c <= '9') return c - '0';
    if (c <= 'F') return c - 'A' + 10;
    if (c <= 'f') return c - 'a' + 10;
    return 0;
}

static byte[] decode(Tokenizer t, int prev_token, int token) {
    byte[] piece = t.vocab[token];
    int offset = 0;
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { offset++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte

    if (piece[0 + offset] == '<' &&
        piece[1 + offset] == '0' &&
        piece[2 + offset] == 'x' &&
        piece[5 + offset] == '>') {
        int byte_val = hex2int(piece[3 + offset]) * 16 + hex2int(piece[4 + offset]);
        return t.byte_pieces[byte_val];
    }
    if (offset == 0) {
        return piece;
    }
    if (t.slided[token] == null) {
        t.slided[token] = Arrays.copyOfRange(piece, 1, piece.length);
    }
    return t.slided[token];
}

static void safe_printf(byte[] piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == null) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        byte byte_val = piece[0];
        if (Character.isISOControl(byte_val) && byte_val != ' ' && byte_val != '\n') {
            return;
        }
    }
    int pos = 0;
    for (; pos < piece.length; ++pos) {
        if (piece[pos] == 0) break;
    }
    System.out.print(new String(piece, 0, pos));
}

static int str_lookup(byte[] str, TokenIndex[] sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    var key = new TokenIndex();
    key.str = str;
    int pos = Arrays.binarySearch(sorted_vocab, key);
    return pos < 0 ? -1 : sorted_vocab[pos].id;
}

static int encode(Tokenizer t, byte[] text, int bos, int eos, int[] tokens ) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == null)  throw new RuntimeException("cannot encode NULL text");

    if (t.sorted_vocab == null) {
        // lazily malloc and sort the vocabulary
        t.sorted_vocab = new TokenIndex[t.vocab_size];
        for (int i = 0; i < t.vocab_size; i++) {
            t.sorted_vocab[i] = new TokenIndex();
            t.sorted_vocab[i].str = t.vocab[i];
            t.sorted_vocab[i].id = i;
        }
        Arrays.sort(t.sorted_vocab);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    byte[] str_buffer = new byte[t.max_token_length*2 +1 +2];
    int str_len = 0;

    // start at 0 tokens
    int n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos != 0) tokens[n_tokens++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(new byte[]{(byte)' ', (byte)0}, t.sorted_vocab, t.vocab_size);
        tokens[n_tokens++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point 竊・UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (int c = 0; text[c] != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((text[c] & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = text[c]; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((text[c+1] & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t.sorted_vocab, t.vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[n_tokens++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[n_tokens++] = str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < n_tokens-1; i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            int pos = 0;
            for (byte b: t.vocab[tokens[i]]) {
                if (b == 0) break;
                str_buffer[pos++] = b;
            }
            for (byte b: t.vocab[tokens[i+1]]) {
                if (b == 0) break;
                str_buffer[pos++] = b;
            }
            str_buffer[pos] = 0;
            int id = str_lookup(str_buffer, t.sorted_vocab, t.vocab_size);
            if (id != -1 && t.vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t.vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < n_tokens-1; i++) {
            tokens[i] = tokens[i+1];
        }
        n_tokens--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos != 0) tokens[n_tokens++] = 2;

    //free(str_buffer);
    return n_tokens;
}


// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

static class ProbIndex implements Comparable<ProbIndex>{
    float prob;
    int index;

    @Override
    public int compareTo(ProbIndex o) {
        if (this.prob > o.prob) return -1;
        if (this.prob < o.prob) return 1;
        return 0;
    }
} // struct used when sorting probabilities during top-p sampling

static class Sampler {
    int vocab_size;
    ProbIndex[] probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    long rng_state[];

    void build(int vocab_size, float temperature, float topp, long rng_seed) {
        this.vocab_size = vocab_size;
        this.temperature = temperature;
        this.topp = topp;
        this.rng_state = new long[]{rng_seed};
        // buffer only used with nucleus sampling; may not need but it's ~small
        this.probindex = new ProbIndex[vocab_size];
        for (int i = 0; i < vocab_size; ++i) {
            this.probindex[i] = new ProbIndex();
        }
    }

    void free() {
        //free(sampler.probindex);
    }
}

static int sample_argmax(float[] probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

static int sample_mult(float[] probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

static int sample_topp(float[] probabilities, int n, float topp, ProbIndex[] probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    final float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    Arrays.sort(probindex);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

static int random_u32(long[] state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    state[0] ^= state[0] >> 12;
    state[0] ^= state[0] << 25;
    state[0] ^= state[0] >> 27;
    return (int)((state[0] * 0x2545F4914F6CDD1Dl) >> 32);
}
static float random_f32(long[] state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

static int sample(Sampler sampler, float[] logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler.temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler.vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler.vocab_size; q++) {
            logits[q] /= sampler.temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler.vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(sampler.rng_state);
        // we sample from this distribution to get the next token
        if (sampler.topp <= 0 || sampler.topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler.vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler.vocab_size, sampler.topp, sampler.probindex, coin);
        }
    }
    return next;
}


// ----------------------------------------------------------------------------
// utilities: time

static long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    return System.currentTimeMillis();
}

// ----------------------------------------------------------------------------
// generation loop

static void generate(Transformer transformer, Tokenizer tokenizer, Sampler sampler, String prompt, int steps) {
    String empty_prompt = "";
    if (prompt == null) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    byte[] pd;
    try {
        pd = prompt.getBytes("utf-8");
    } catch (UnsupportedEncodingException ex) {
        throw new UncheckedIOException(ex);
    }
    byte[] prompt_data = new byte[pd.length + 1];
    for (int i = 0; i < pd.length; ++i) {
        prompt_data[i] = pd[i];
    }
    prompt_data[pd.length] = 0;
    int[] prompt_tokens = new int[prompt_data.length + 3]; // +3 for '\0', ?BOS, ?EOS
    int num_prompt_tokens = encode(tokenizer, prompt_data, 1, 0, prompt_tokens);
    if (num_prompt_tokens < 1) {
        System.err.printf("something is wrong, expected at least 1 prompt token\n");
        System.exit(1);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float[] logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        byte[] piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        //fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    System.out.printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        System.err.printf("achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    //free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.
/*
static void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

static void chat(Transformer transformer, Tokenizer tokenizer, Sampler sampler,
          String cli_user_prompt, String cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char[] system_prompt = new char[512];
    char[] user_prompt = new char[512];
    char[] rendered_prompt = new char[1152];
    int num_prompt_tokens = 0;
    int[] prompt_tokens = new int[1152];
    int user_idx;

    // start the main loop
    boolean user_turn = true; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = false;
            System.out.printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = true; }

        // forward the transformer to get logits for the next token
        MemorySegment logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { System.out.printf("\n"); }
    }
    System.out.printf("\n");
    free(prompt_tokens);
}
*/

// ----------------------------------------------------------------------------
// CLI, include only if not testing

static void error_usage() {
    System.err.printf("Usage:   run <checkpoint> [options]\n");
    System.err.printf("Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    System.err.printf("Options:\n");
    System.err.printf("  -t <float>  temperature in [0,inf], default 1.0\n");
    System.err.printf("  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    System.err.printf("  -s <int>    random seed, default time(NULL)\n");
    System.err.printf("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    System.err.printf("  -i <string> input prompt\n");
    System.err.printf("  -z <string> optional path to custom tokenizer\n");
    System.err.printf("  -m <string> mode: generate|chat, default: generate\n");
    System.err.printf("  -y <string> (optional) system prompt in chat mode\n");
    System.err.printf("  -v <string> 'on' if vectorize\n");
    System.exit(1);
}

public static void main(String[] args) {
    // default parameters
    String checkpoint_path = d260k;  // e.g. out/model.bin
    String tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    String prompt = "Momotaro is born in peatch, ";        // prompt string
    long rng_seed = 1234; // seed rng with time by default
    String mode = "generate";    // generate|chat
    String system_prompt = null; // the (optional) system prompt to use in chat mode

    int argc = args.length;
    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 1) {
        checkpoint_path = args[0];
    } else {
        // error_usage();
    }
    for (int i = 1; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (args[i].charAt(0) != '-') { error_usage(); } // must start with dash
        if (args[i].length() != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        switch (args[i].charAt(1)) {
          case 't' -> temperature = Float.parseFloat(args[i + 1]);
          case 'p' -> topp = Float.parseFloat(args[i + 1]);
          case 's' -> rng_seed = Integer.parseInt(args[i + 1]);
          case 'n' -> steps = Integer.parseInt(args[i + 1]);
          case 'i' -> prompt = args[i + 1];
          case 'z' -> tokenizer_path = args[i + 1];
          case 'm' -> mode = args[i + 1];
          case 'y' -> system_prompt = args[i + 1];
          case 'v' -> vectorize = "on".equalsIgnoreCase(args[i + 1]);
          default -> error_usage();
        }
    }

    System.out.println("vectorize: " + (vectorize ? "on" : "off"));

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = System.currentTimeMillis();
    if (temperature < 0.0) temperature = 0.0f;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9f;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer = new Transformer();
    transformer.build(checkpoint_path);
    System.out.println(transformer.config);

    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer = new Tokenizer();

    tokenizer.build(tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler = new Sampler();
    sampler.build(transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (mode.equals("generate")) {
        generate(transformer, tokenizer, sampler, prompt, steps);
    } else if (mode.equals("chat")) {
        //chat(transformer, tokenizer, sampler, prompt, system_prompt, steps);
        System.out.println("chat mode is not implemented yet");
    } else {
        System.err.printf("unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    sampler.free();
    tokenizer.free();
    transformer.free();
}



public static void main_(String[] args) {
    test_build();
    System.out.println("fin");
}

static String d260k = "stories260K.bin";
public static void test_build() {
    var transformer = new Transformer();
    transformer.build(d260k);
    System.out.println(transformer.config);
    transformer.free();
}

public static void test_transformer_read(String[] args) throws IOException {
    var transformer = new Transformer();
    transformer.read_checkpoint(d260k);
    System.out.println(transformer.config);
}

public static void test_config_load(String[] args) {
    var data = Arena.ofShared().allocateFrom(ValueLayout.JAVA_INT, IntStream.range(0, 14).toArray());
    var c1 = new Config();
    var alloc = SegmentAllocator.slicingAllocator(data);
    c1.load(alloc);
    var c2 = new Config();
    c2.load(alloc);
    System.out.println(c1.dim);
    System.out.println(c1.hidden_dim);
    System.out.println(c2.dim);
    System.out.println(c2.hidden_dim);

}

}

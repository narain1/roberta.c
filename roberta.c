#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include "Tensor.h"

#define MAX_TOKENS 1000
#define MAX_LENGTH 100

typedef struct 
{
  char *s;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512];
} Tokenizer;

struct ModelConfig {
  unsigned int vocab_size;
  unsigned int hidden_size;
  unsigned int intermediate_size;
  unsigned int n_max_tokens;
  unsigned int n_att_heads;
  unsigned int n_hidden_layers;
};

struct ModelConfig base_config = {
  .hidden_size = 768,
  .intermediate_size = 3072,
  .n_hidden_layers = 12,
  .n_att_heads = 12,
  .vocab_size = 30522,
  .n_max_tokens = 512
};

struct EmbeddingLayer {
  struct Tensor *word_emb;
  struct Tensor *pos_emb;
  struct Tensor *tok_type_w;
  struct LayerNorm *ln;
};

struct EncoderLayer {
  struct Linear *query;
  struct Linear *key;
  struct Linear *value;
  struct Linear *output;
  struct LayerNorm *ln1;
  struct Linear *ff_in;
  struct Linear *ff_out;
  struct LayerNorm *ln2;
};

struct Classifier {
  struct Tensor *pred_bias;
  struct Linear *transform_linear;
  struct LayerNorm *ln;
  struct Tensor *decoder_weight;
  struct Linear *seq;
};

struct RobertaModel {
  struct ModelConfig *config;
  struct EmbeddingLayer *embed;
  struct EncoderLayer *layers;
  struct Linear *pool;
  struct Classifier *clf;
};

struct Buffer {
  struct Tensor *word_emb;
  struct Tensor *pos_emb;
  struct Tensor *tok_type_w;
  struct Tensor *layer_q;
  struct Tensor *layer_k;
  struct Tensor *layer_v;
};

void load_tensor(float *data, const unsigned long size, unsigned long *offset, char *buffer) {
  data = (float*)malloc(sizeof(float) * size);
  printf("size of offset is %lu %lu\n", *offset, size);
  for (size_t i=0; i<size; i++) {
    data[i] = *((float *)(buffer + *offset + i * sizeof(float)));
  }
  *offset += size; 
}

void load_config(struct ModelConfig *data, char *buffer, int *conf_sz) {
  unsigned int *conf_arr = (int *)malloc(sizeof(int) * *conf_sz);
  for (int i=0; i< *conf_sz; i++) {
    conf_arr[i] = *((unsigned int*)(buffer + sizeof(int) + sizeof(int) * i));
  }

  // initialize config
  data->vocab_size = conf_arr[0];
  data->hidden_size = conf_arr[1];
  data->intermediate_size = conf_arr[2];
  data->n_max_tokens = conf_arr[3];
  data->n_att_heads = conf_arr[4];
  data->n_hidden_layers = conf_arr[5];
}

void initialize_parameters(struct RobertaModel *model) {

  // initializing embedding layer
  model->embed = (struct EmbeddingLayer*)malloc(sizeof(struct EmbeddingLayer));
  model->embed->word_emb = (struct Tensor*)malloc(sizeof(struct Tensor));
  model->embed->pos_emb = (struct Tensor*)malloc(sizeof(struct Tensor));
  model->embed->tok_type_w = (struct Tensor*)malloc(sizeof(struct Tensor));
  initialize_ln(&(model->embed->ln));
  
  int n = model->config->n_hidden_layers;

  struct EncoderLayer *layers = (struct EncoderLayer*)malloc(n * sizeof(struct EncoderLayer));

  for (int i=0; i<n; i++) {
    initialize_linear(&(layers[i].query));
    initialize_linear(&(layers[i].key));
    initialize_linear(&(layers[i].value));
    initialize_linear(&(layers[i].output));
    initialize_ln(&(layers[i].ln1));
    initialize_linear(&(layers[i].ff_in));
    initialize_linear(&(layers[i].ff_out));
    initialize_ln(&(layers[i].ln2));
  }

  model->layers = layers;
  initialize_linear(&(model->pool));
  model->clf = (struct Classifier*)malloc(sizeof(struct Classifier));
  model->clf->pred_bias = (struct Tensor*)malloc(sizeof(struct Tensor));
  initialize_linear(&(model->clf->transform_linear));
  initialize_ln(&(model->clf->ln));
  model->clf->decoder_weight = (struct Tensor*)malloc(sizeof(struct Tensor));
  initialize_linear(&(model->clf->seq));
}

void initialize_buffer(struct Buffer *buffer) {
  buffer->word_emb = (struct Tensor*)malloc(sizeof(struct Tensor));
  buffer->pos_emb = (struct Tensor*)malloc(sizeof(struct Tensor));
  buffer->tok_type_w = (struct Tensor*)malloc(sizeof(struct Tensor));
  buffer->layer_q = (struct Tensor*)malloc(sizeof(struct Tensor));
  buffer->layer_k = (struct Tensor*)malloc(sizeof(struct Tensor));
  buffer->layer_v = (struct Tensor*)malloc(sizeof(struct Tensor));
}

void load_model(struct RobertaModel *model, const char *fname) {
  printf("%s: loading model from '%s'\n", __func__, fname);
  
  int fd;
  char *buffer;
  float *arr;
  off_t file_size;
  unsigned long cur_size = 0;
  unsigned long offset = 0;

  // sanity check file
  fd = open("model.bin", O_RDONLY);
  if (fd == -1) { perror("error opening file"); exit(EXIT_FAILURE); }
  printf("file opened\n");

  // get file size to load in memory
  file_size = lseek(fd, 0, SEEK_END);
  if (file_size == -1) { fprintf(stderr, "error getting file size\n"), close(fd); exit(EXIT_FAILURE); }
  printf("model size = %lu\n", file_size);

  buffer = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (buffer == MAP_FAILED) { printf("error mapping file to memory"); close(fd); exit(EXIT_FAILURE); }
  close(fd);
  
  int *conf_sz = (int*)buffer;
  // printf("config size %d\n", *conf_sz);

  int *conf_arr = (int *)malloc(sizeof(int) * *conf_sz);
  model->config = (struct ModelConfig*)malloc(sizeof(struct ModelConfig));
 
  load_config(model->config, buffer, conf_sz);
  initialize_parameters(model);

  offset += sizeof(int) * (*conf_sz + 1);
  load_tensor2d(model->embed->word_emb, 
                model->config->vocab_size,
                model->config->hidden_size,
                &offset,
                buffer);
  
  load_tensor2d(model->embed->pos_emb,
                model->config->n_max_tokens,
                model->config->hidden_size,
                &offset,
                buffer);

  load_tensor2d(model->embed->tok_type_w,
                2,
                model->config->hidden_size,
                &offset, buffer);


  load_ln(model->embed->ln, model->config->hidden_size, &offset, buffer);

  int n_layers = model->config->n_hidden_layers;

  for (int i=0; i<n_layers; i++) {
    load_linear(model->layers[i].query, model->config->hidden_size, model->config->hidden_size, &offset, buffer);
    load_linear(model->layers[i].key, model->config->hidden_size, model->config->hidden_size, &offset, buffer);
    load_linear(model->layers[i].value, model->config->hidden_size, model->config->hidden_size, &offset, buffer);
    load_linear(model->layers[i].output, model->config->hidden_size, model->config->hidden_size, &offset, buffer);

    load_ln(model->layers[i].ln1, model->config->hidden_size, &offset, buffer);

    load_linear(model->layers[i].ff_in, model->config->intermediate_size, model->config->hidden_size, &offset, buffer);
    load_linear(model->layers[i].ff_out, model->config->hidden_size, model->config->intermediate_size, &offset, buffer);

    load_ln(model->layers[i].ln2, model->config->hidden_size, &offset, buffer);
  }

  load_linear(model->pool, model->config->hidden_size, model->config->hidden_size, &offset, buffer);
  // printf("loaded layers, \n");
  // printf("model size     %lu\n", file_size);
  // printf("offset current %lu\n", offset);
  load_tensor1d(model->clf->pred_bias, model->config->vocab_size, &offset, buffer);
  // printf("loaded bias \n");
  // printf("model size     %lu\n", file_size);
  // printf("offset current %lu\n", offset);
  load_linear(model->clf->transform_linear, model->config->hidden_size, model->config->hidden_size, &offset, buffer);
  // printf("loaded linear \n");
  load_ln(model->clf->ln, model->config->hidden_size, &offset, buffer);
  load_tensor2d(model->clf->decoder_weight, model->config->vocab_size, model->config->hidden_size, &offset, buffer);
  load_linear(model->clf->seq, 2, model->config->hidden_size, &offset, buffer);
  // printf("model size     %lu\n", file_size);
  // printf("offset current %lu\n", offset);
  if (file_size == offset) {
    printf("model loaded successfully\n");
  }
}

void free_encoder_layers(struct EncoderLayer *layers, int n_layers) {
    if (layers != NULL) {
        for (int i = 0; i < n_layers; ++i) {
            free_linear(layers[i].query);
            free_linear(layers[i].key);
            free_linear(layers[i].value);
            free_linear(layers[i].output);
            free_ln(layers[i].ln1);
            free_linear(layers[i].ff_in);
            free_linear(layers[i].ff_out);
            free_ln(layers[i].ln2);
        }
        free(layers);
    }
}

void free_embedding_layer(struct EmbeddingLayer *embed) {
    if (embed != NULL) {
        free_tensor(embed->word_emb);
        free_tensor(embed->pos_emb);
        free_tensor(embed->tok_type_w);
        free_ln(embed->ln);
        free(embed);
    }
}

void free_classifier(struct Classifier *clf) {
    if (clf != NULL) {
        free_tensor(clf->pred_bias);
        free_linear(clf->transform_linear);
        free_ln(clf->ln);
        free_tensor(clf->decoder_weight);
        free_linear(clf->seq);
        free(clf); 
    }
}

void free_model(struct RobertaModel *model) {
    if (model != NULL) {
       // if (model->config != NULL) {
       //     free(model->config); 
       // }
        free_embedding_layer(model->embed);
        free_encoder_layers(model->layers, model->config->n_hidden_layers);
        free_classifier(model->clf);
        free_linear(model->pool);
        free(model); 
    }
}

void build_tokenizer(Tokenizer *t, char* tokenizer_path, int vocab_size) {
  t->vocab_size = vocab_size;
  t->vocab = (char**)malloc(vocab_size * sizeof(char*));
  t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL;

  for (int i=0; i<256; i++) {
    t->byte_pieces[i*2] = (unsigned char)i;
    t->byte_pieces[i*2+1] = '\0';
  }

  FILE *file = fopen(tokenizer_path, "rb");
  if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
  if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
  printf("max token length %d\n", t->max_token_length);
  int len;
  for (int i=0; i<vocab_size; i++) {
    if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    if (fread(&len, sizeof(int), 1, file) != 1) {fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    t->vocab[i] = (char *)malloc(len + 1);
    if (fread(t->vocab[i], len, 1, file) != 1) {fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    t->vocab[i][len] = '\0';
  }
  fclose(file);
}

void free_tokenizer(Tokenizer* t) {
  for (int i=0; i<t->vocab_size; i++) { free(t->vocab[i]); }
  free(t->vocab);
  free(t->vocab_scores);
  free(t->sorted_vocab);

  printf("tokenizer freed\n");
}

void print_model_tensors(const struct RobertaModel *model) {
    // Print embedding layer tensors
    print_tensor_shape("Word Embeddings", model->embed->word_emb);
    print_tensor_shape("Position Embeddings", model->embed->pos_emb);
    print_tensor_shape("Token Type Embeddings", model->embed->tok_type_w);
    print_tensor_shape("Embedding LayerNorm Gamma", model->embed->ln->gamma);
    print_tensor_shape("Embedding LayerNorm Beta", model->embed->ln->beta);

    // Print encoder layer tensors
    for (int i = 0; i < model->config->n_hidden_layers; ++i) {
        char layer_prefix[64];
        sprintf(layer_prefix, "Encoder Layer %d", i);

        char tensor_name[128];

        sprintf(tensor_name, "%s Query Weight", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].query->w);
        sprintf(tensor_name, "%s Query Bias", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].query->b);

        sprintf(tensor_name, "%s Key Weight", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].key->w);
        sprintf(tensor_name, "%s Key Bias", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].key->b);

        sprintf(tensor_name, "%s Value Weight", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].value->w);
        sprintf(tensor_name, "%s Value Bias", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].value->b);

        sprintf(tensor_name, "%s Output Dense Weight", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].output->w);
        sprintf(tensor_name, "%s Output Dense Bias", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].output->b);

        sprintf(tensor_name, "%s LayerNorm1 Gamma", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].ln1->gamma);
        sprintf(tensor_name, "%s LayerNorm1 Beta", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].ln1->beta);

        sprintf(tensor_name, "%s FFN Dense In Weight", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].ff_in->w);
        sprintf(tensor_name, "%s FFN Dense In Bias", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].ff_in->b);

        sprintf(tensor_name, "%s FFN Dense Out Weight", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].ff_out->w);
        sprintf(tensor_name, "%s FFN Dense Out Bias", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].ff_out->b);

        sprintf(tensor_name, "%s LayerNorm2 Gamma", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].ln2->gamma);
        sprintf(tensor_name, "%s LayerNorm2 Beta", layer_prefix);
        print_tensor_shape(tensor_name, model->layers[i].ln2->beta);
    }

    // Print classifier layer tensors
    print_tensor_shape("Classifier Pred Bias", model->clf->pred_bias);
    print_tensor_shape("Classifier Transform Linear Weight", model->clf->transform_linear->w);
    print_tensor_shape("Classifier Transform Linear Bias", model->clf->transform_linear->b);
    print_tensor_shape("Classifier LayerNorm Gamma", model->clf->ln->gamma);
    print_tensor_shape("Classifier LayerNorm Beta", model->clf->ln->beta);
    print_tensor_shape("Classifier Decoder Weight", model->clf->decoder_weight);
    print_tensor_shape("Classifier Seq Weight",model->clf->seq->w);
    print_tensor_shape("Classifier Seq Bias", model->clf->seq->b);
}

int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex*)a)->s, ((TokenIndex*)b)->s);
}

int str_lookup(char *s, TokenIndex *sorted_vocab, int vocab_size) {
  TokenIndex tok = { .s = s };
  TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
  return res != NULL ? res->id : -1;
}

void bpe_encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  if (t->sorted_vocab == NULL) {
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i=0; i<t->vocab_size; i++) {
      t->sorted_vocab[i].s = t->vocab[i];
      t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  char *str_buffer = malloc((t->max_token_length*2+1) * sizeof(char));
  *n_tokens = 0;
  if (bos) tokens[(*n_tokens)++] = 0;
  for (char *c = text; *c != '\0'; c++) {
    sprintf(str_buffer, "%c", *c);
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
    if (id == -1) { printf("not good\n"); exit(1);}
    tokens[*n_tokens] = id;
    (*n_tokens)++;
  }

  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i=0; i<(*n_tokens-1); i++) {
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }
    if (best_idx == -1) {
      break;
    }

    tokens[best_idx] = best_id;
    for (int i=best_idx+1; i < (*n_tokens-1); i++) {
      tokens[i] = tokens[i+1];
    }
    (*n_tokens)--;
  }

  free(str_buffer);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void ln_forward(struct LayerNorm *ln, struct Tensor *inp, unsigned int axis, float eps)
{
  if (axis >= inp->ndim) {
    printf("Error exis %u is out of bounds for tensor with %u dimensions. \n", axis, inp->ndim);
    return;
  }

  struct Tensor mean_tensor = reduce_mean_axis(inp, axis);
  printf("mean computed\n");
  print_tensor(&mean_tensor);
  struct Tensor std_tensor = reduce_std_axis(inp, axis);
  print_tensor(&std_tensor);
  printf("std computed\n");

  for (unsigned long i=0; i<inp->size; ++i) {
    int idx[4] = {0};
    int temp = i;

    for (int d=inp->ndim - 1; d >= 0; --d) {
      idx[d] = temp % inp->shape[d];
      temp /= inp->shape[d];
    }

    int result_idx = 0;
    for (unsigned int d=0; d < inp->ndim; ++d) {
      if (d != axis) {
        result_idx += result_idx * inp->shape[d] + idx[d];
      }
    }

    float mean = mean_tensor.data[result_idx];
    float std = std_tensor.data[result_idx];
    float norm = (inp->data[i] - mean) / (std + eps);

    unsigned long gamma_idx = idx[axis];
    inp->data[i] = ln->gamma->data[gamma_idx] * norm + ln->beta->data[gamma_idx];
  }

  free_tensor(&mean_tensor);
  free_tensor(&std_tensor);
}

void linear_forward(struct Linear *l, struct Tensor *x, struct Tensor *o)
{
  struct Tensor temp;
  transpose_tensor(x, &temp);

  if (temp.shape[x->ndim - 1] != l->w->shape[0]) {
    printf("error: Input dimensions do not match weight dimensions. \n");
    return;
  }

  mm_f32(&temp, l->w, o);
  free_tensor(&temp);
  print_tensor_shape("mm intermediate", o);
  _sum_tensors_broadcast(o, l->b);
}
  

void forward(struct RobertaModel *model, int *tokens, int n_tokens) {
  struct Buffer *buffer = (struct Buffer*)malloc(sizeof(struct Buffer));
  buffer->word_emb = (struct Tensor*)malloc(sizeof(struct Tensor));
  buffer->pos_emb = (struct Tensor*)malloc(sizeof(struct Tensor));
  buffer->tok_type_w = (struct Tensor*)malloc(sizeof(struct Tensor));
  buffer->layer_q = (struct Tensor*)malloc(sizeof(struct Tensor)); 
  buffer->layer_k = (struct Tensor*)malloc(sizeof(struct Tensor)); 
  buffer->layer_v = (struct Tensor*)malloc(sizeof(struct Tensor)); 
  // load word embedding
  map_embeddings(
      buffer->word_emb, 
      model->embed->word_emb, 
      model->config->hidden_size,
      tokens,
      n_tokens);
  // load positional embedding
  int *index;
  arange(&(index), 0, n_tokens);

  map_embeddings(
      buffer->pos_emb,
      model->embed->pos_emb,
      model->config->hidden_size,
      index,
      n_tokens);

  _sum_tensors(buffer->word_emb, buffer->pos_emb);

  int *tok_index;
  arr_zeros(&(tok_index), n_tokens);
  map_embeddings(
    buffer->tok_type_w,
    model->embed->tok_type_w,
    model->config->hidden_size,
    tok_index,
    n_tokens);

  _sum_tensors(buffer->word_emb, buffer->tok_type_w);

  ln_forward(model->embed->ln, buffer->word_emb, 1, 1.0f/100000);
  // for (int n=0; n<model->config->n_hidden_layers; n++) {
  for (int n=0; n<1; n++) {
    print_tensor_shape("query weight_shape", model->layers[n].query->w);
    print_tensor_shape("query_input_shape", buffer->word_emb);
    linear_forward(model->layers[n].query, buffer->word_emb, buffer->layer_q);
    linear_forward(model->layers[n].key, buffer->word_emb, buffer->layer_k);
    linear_forward(model->layers[n].value, buffer->word_emb, buffer->layer_v);
  }
  print_tensor(buffer->layer_q);
}

void main() {
  unsigned int vocab_size = 30522;
  Tokenizer tokenizer;
  char *tokenizer_path = "tokenizer.bin";
  char *model_path = "model.bin";
  build_tokenizer(&tokenizer, tokenizer_path, vocab_size);
  printf("loaded tokenizer\n");
  char *prompt = "Buying a mattress online can be a waking nightmare, ";
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int*)malloc((strlen(prompt)+3)*sizeof(int));
  bpe_encode(&tokenizer, prompt, 0, 2, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "failed here\n");
    exit(EXIT_FAILURE);
  }

  for (int i=0; i<num_prompt_tokens; i++) {
    printf("%d, ", prompt_tokens[i]);
  }
  printf("\n");
  printf("\nThe number of tokens is %d\n", num_prompt_tokens);

  for (int i=0; i<num_prompt_tokens; i++) {
    char *token = decode(&tokenizer, prompt_tokens[i], prompt_tokens[i]);
    printf("%s ", token);
  }
  printf("\n");

  struct RobertaModel *model = malloc(sizeof(struct RobertaModel));
    
  printf("\n\n\n\n");
  load_model(model, "model.bin"); 

  forward(model, prompt_tokens, num_prompt_tokens);

  // print_model_tensors(model);

  printf("execution completed\n");
  free_tokenizer(&tokenizer);
  free_model(model);
  printf("freed tokenizer\n");
}

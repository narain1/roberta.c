#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#define MAX_TOKENS 1000
#define MAX_LENGTH 100

typedef struct {
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
  int vocab_size;
  int hidden_size;
  int intermediate_size;
  int n_max_tokens;
  int n_att_heads;
  int n_hidden_layers;
};

struct ModelConfig base_config = {
  .hidden_size = 768,
  .intermediate_size = 3072,
  .n_hidden_layers = 12,
  .n_att_heads = 12,
  .vocab_size = 30522,
  .n_max_tokens = 512
};

struct Tensor {
  float *data;
  int *shape;
};

struct Linear {
  struct Tensor *w;
  struct Tensor *b;
};

struct LayerNorm {
  struct Tensor *gamma;
  struct Tensor *beta;
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
  struct Linear *ff_in;
  struct Linear *ff_out;
  struct LayerNorm *ln;
};

struct RobertaModel {
  struct ModelConfig *config;
  struct EmbeddingLayer *embed;
  struct EncoderLayer *layers;
};

size_t reduce_product(int *arr, int size) {
    size_t product = 1;
    for (int i = 0; i < size; i++) {
        product *= (size_t)arr[i];
    }
    return product;
}

void load_tensor(float *data, const unsigned long size, unsigned long *offset, char *buffer) {
  printf("in\n");
  data = (float*)malloc(sizeof(float) * size);
  printf("size of offset is %lu %lu\n", *offset, size);
  for (size_t i=0; i<size; i++) {
    data[i] = *((float *)(buffer + *offset + i * sizeof(float)));
  }
  *offset += size; 
}

void load_config(struct ModelConfig *data, char *buffer, int *conf_sz) {
  int *conf_arr = (int *)malloc(sizeof(int) * *conf_sz);
  for (int i=0; i< *conf_sz; i++) {
    conf_arr[i] = *((int*)(buffer + sizeof(int) + sizeof(int) * i));
  }

  // initialize config
  data->vocab_size = conf_arr[0];
  data->hidden_size = conf_arr[1];
  data->intermediate_size = conf_arr[2];
  data->n_max_tokens = conf_arr[3];
  data->n_att_heads = conf_arr[4];
  data->n_hidden_layers = conf_arr[5];
}

void initialize_linear(struct Linear *layer) {
  layer = (struct Linear*)malloc(sizeof(struct Linear));
  layer->w = (struct Tensor*)malloc(sizeof(struct Tensor));
  layer->b = (struct Tensor*)malloc(sizeof(struct Tensor));
}

void initalize_ln(struct LayerNorm *ln) {
  ln = (struct LayerNorm*)malloc(sizeof(struct LayerNorm));
  ln->gamma = (struct Tensor*)malloc(sizeof(struct Tensor));
  ln->beta = (struct Tensor*)malloc(sizeof(struct Tensor));
}


void initialize_parameters(struct RobertaModel *model) {

  // initializing embedding layer
  model->embed = (struct EmbeddingLayer*)malloc(sizeof(struct EmbeddingLayer));
  model->embed->word_emb = (struct Tensor*)malloc(sizeof(struct Tensor));
  model->embed->pos_emb = (struct Tensor*)malloc(sizeof(struct Tensor));
  model->embed->tok_type_w = (struct Tensor*)malloc(sizeof(struct Tensor));
  initalize_ln(model->embed->ln);
  
  int n = model->config->n_hidden_layers;

  model->layers = (struct EncoderLayer*)malloc(n * sizeof(struct EncoderLayer));
  model->layers = (struct EncoderLayer*)malloc(sizeof(struct EncoderLayer));

  initialize_linear(model->layers->query);
  initialize_linear(model->layers->key);
  initialize_linear(model->layers->value);
  initialize_linear(model->layers->ff_in);
  initialize_linear(model->layers->ff_out);
  initalize_ln(model->layers->ln);

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

  buffer = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (buffer == MAP_FAILED) { printf("error mapping file to memory"); close(fd); exit(EXIT_FAILURE); }
  close(fd);
  
  int *conf_sz = (int*)buffer;
  printf("config size %d\n", *conf_sz);

  int *conf_arr = (int *)malloc(sizeof(int) * *conf_sz);
  model->config = (struct ModelConfig*)malloc(sizeof(struct ModelConfig));
 
  load_config(model->config, buffer, conf_sz);
  initialize_parameters(model);

  cur_size = model->config->vocab_size * model->config->hidden_size;
  offset += sizeof(int) * (*conf_sz + 1);
  load_tensor(model->embed->word_emb->data, cur_size, &offset, buffer);
  
  cur_size = model->config->n_max_tokens * model->config->hidden_size;
  load_tensor(model->embed->pos_emb->data, cur_size, &offset, buffer);

  cur_size = 2 * model->config->hidden_size;
  load_tensor(model->embed->tok_type_w->data, cur_size, &offset, buffer);

  cur_size = model->config->hidden_size;
  load_tensor(model->embed->ln->gamma->data, cur_size, &offset, buffer);

  cur_size = model->config->hidden_size;
  load_tensor(model->embed->ln->beta->data, cur_size, &offset, buffer);

  int n_layers = model->config->n_hidden_layers;

  cur_size = model->config->hidden_size * model->config->hidden_size;
  load_tensor(model->layers->query->w->data, cur_size, &offset, buffer);

   // cur_size = model->config->hidden_size;
    //load_tensor(model->layers->query[i]->b->data, cur_size, &offset, buffer);

    //cur_size = model->config->hidden_size * model->config->hidden_size;
    //load_tensor(model->layers->key[i]->w->data, cur_size, &offset, buffer);

    //cur_size = model->config->hidden_size;
    //load_tensor(model->layers->key[i]->b->data, cur_size, &offset, buffer);

    //cur_size = model->config->hidden_size * model->config->hidden_size;
    //load_tensor(model->layers->value[i]->w->data, cur_size, &offset, buffer);
 
    //cur_size = model->config->hidden_size;
    //load_tensor(model->layers->key[i]->b->data, cur_size, &offset, buffer);


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




void main() {
  unsigned int vocab_size = 30522;
  Tokenizer tokenizer;
  char *tokenizer_path = "tokenizer.bin";
  char *model_path = "model.bin";
  build_tokenizer(&tokenizer, tokenizer_path, vocab_size);
  printf("loaded tokenizer\n");
  char *prompt = "Buying a mattress online can be a waking nightmare, and picking the wrong one can literally cause bad dreams or kill your back. It doesn't help that the online market is flooded with options or that there are more dedicated mattress review sites than stars in the sky.";
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
    
  load_model(model, "model.bin"); 
  free_tokenizer(&tokenizer);
  printf("freed tokenizer\n");
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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


void main() {
  Tokenizer tokenizer;
  uint32_t vocab_size = 50265;
  char *tokenizer_path = "tokenizer.bin";
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
    
  printf("token encoding done\n");
  free_tokenizer(&tokenizer);
  printf("freed tokenizer");
}

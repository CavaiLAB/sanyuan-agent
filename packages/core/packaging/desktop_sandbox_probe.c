/* Native filesystem behavior probe for the local Codex sandbox.
 *
 * This binary intentionally depends only on the platform C runtime. A
 * PyInstaller one-file executable cannot be used here because its extraction
 * step requires a writable temporary directory before the probe can run.
 */
#include <errno.h>
#include <stdio.h>
#include <string.h>

static int valid_nonce(const char *value) {
    size_t index;
    if (strlen(value) != 32) {
        return 0;
    }
    for (index = 0; index < 32; ++index) {
        const char current = value[index];
        if (!((current >= '0' && current <= '9') ||
              (current >= 'a' && current <= 'f'))) {
            return 0;
        }
    }
    return 1;
}

static void attempt_write(const char *path, const char *nonce, const char *label) {
    FILE *stream;
    errno = 0;
    stream = fopen(path, "wb");
    if (stream == NULL) {
        if (errno == EACCES || errno == EPERM) {
            printf("ICLAW_PROBE_%s_%s_DENIED\n", nonce, label);
        } else {
            printf("ICLAW_PROBE_%s_%s_ERR_%d\n", nonce, label, errno);
        }
        return;
    }
    fclose(stream);
    printf("ICLAW_PROBE_%s_%s_WROTE\n", nonce, label);
}

int main(int argc, char **argv) {
    const char *mode;
    const char *nonce;
    const char *inside;

    if (argc < 5 || strcmp(argv[1], "sandbox-probe") != 0) {
        return 2;
    }
    mode = argv[2];
    nonce = argv[3];
    inside = argv[4];
    if (!valid_nonce(nonce)) {
        return 2;
    }

    printf("ICLAW_PROBE_%s_RAN\n", nonce);
    if (argc == 5 && strcmp(mode, "read-only") == 0) {
        attempt_write(inside, nonce, "READ_ONLY");
        return 0;
    }
    if (argc == 7 && strcmp(mode, "workspace") == 0 &&
        strcmp(argv[5], "--outside") == 0) {
        attempt_write(inside, nonce, "INSIDE");
        attempt_write(argv[6], nonce, "OUTSIDE");
        return 0;
    }
    return 2;
}

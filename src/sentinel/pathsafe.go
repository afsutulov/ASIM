package main

// pathsafe.go — защита от подмены путей (path traversal).
//
// safeName очищает строку для безопасного использования как компонента
// имени файла: оставляет только буквы, цифры и символы -_.
// Любой другой символ (/, \, .., пробел и т.д.) заменяется на _.
// Пустая строка превращается в "_empty".
//
// Используется во всех местах где внешние данные (TileID, Date, Name,
// OutputName, ONNXFile) подставляются в filepath.Join / fmt.Sprintf.

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

var reUnsafe = regexp.MustCompile(`[^A-Za-z0-9\-_.]`)

// safeName возвращает строку, безопасную для использования как имя файла.
// Запрещённые символы заменяются на '_'. Компоненты пути (/ и \) тоже
// заменяются, поэтому результат никогда не выйдет за пределы целевого каталога.
func safeName(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "_empty"
	}
	// Заменяем все небезопасные символы на '_'
	s = reUnsafe.ReplaceAllString(s, "_")
	// Дополнительная защита: убираем ведущие точки (скрытые файлы / "..")
	s = strings.TrimLeft(s, ".")
	if s == "" {
		return "_empty"
	}
	return s
}

// safeJoinName строит filepath.Join(dir, safeName(name)+ext) и проверяет,
// что результат находится внутри dir. Если нет — возвращает ошибку.
func safeJoinName(dir, name, ext string) (string, error) {
	clean := safeName(name) + ext
	target := filepath.Join(dir, clean)
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	absTarget, err := filepath.Abs(target)
	if err != nil {
		return "", err
	}
	if !strings.HasPrefix(absTarget, absDir+string(filepath.Separator)) &&
		absTarget != absDir {
		return "", fmt.Errorf("path traversal detected: %q is outside %q", absTarget, absDir)
	}
	return target, nil
}

import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import type { DecisionResponse } from '../src/types'

test('真实模型三种问题：图表、数值与服务响应一致', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('服务已就绪', { exact: true })).toBeVisible()
  for (const type of ['choice', 'score', 'noul']) {
    await page.getByLabel('快速示例').selectOption(type)
    const responsePromise = page.waitForResponse((response) => response.url().endsWith('/api/v1/decisions'))
    await page.getByRole('button', { name: '运行推理' }).click()
    const response = await responsePromise
    expect(response.status()).toBe(200)
    const data = await response.json() as DecisionResponse
    const answer = data.answers.q1
    expect(answer.type).toBe(type)
    const expected = type === 'choice' ? answer.choice! : type === 'score'
      ? answer.score!.toFixed(3) : `${(answer.noul! * 100).toFixed(1)}%`
    await expect(page.getByTestId('answer-value')).toHaveText(expected)
    const rows = page.getByTestId('probability-row')
    await expect(rows).toHaveCount(Object.keys(answer.probabilities).length)
    const probabilities = Object.values(answer.probabilities)
    for (let i = 0; i < probabilities.length; i++) {
      await expect(rows.nth(i).getByTestId('probability-value')).toHaveText(`${(probabilities[i] * 100).toFixed(1)}%`)
      const width = await rows.nth(i).locator('.probability-fill').evaluate((element) => (element as HTMLElement).style.width)
      expect(parseFloat(width)).toBeCloseTo(probabilities[i] * 100, 3)
    }
    await page.getByText('查看请求与响应 JSON', { exact: true }).click()
    expect(JSON.parse((await page.getByTestId('response-json').textContent())!)).toEqual(data)
    await page.getByText('查看请求与响应 JSON', { exact: true }).click()
  }
  await page.getByLabel('待判断文本').fill('小王负责测试，小李负责发布。')
  await expect(page.getByText('输入已修改。下方为上次输入的结果，请重新运行。')).toBeVisible()
})

test('未连接可恢复，等待真实推理时阻止重复提交', async ({ page }) => {
  await page.route('**/api/health/ready', (route) => route.fulfill({ status: 502, json: { error: 'service_unavailable' } }))
  await page.goto('/')
  await expect(page.getByText('服务未连接', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: '运行推理' })).toBeDisabled()
  await page.unroute('**/api/health/ready')
  await page.getByRole('button', { name: '重新连接' }).click()
  await expect(page.getByText('服务已就绪', { exact: true })).toBeVisible()
  let release!: () => void
  const gate = new Promise<void>((resolve) => { release = resolve })
  let requests = 0
  await page.route('**/api/v1/decisions', async (route) => {
    requests++
    await gate
    await route.continue()
  })
  await page.getByRole('button', { name: '运行推理' }).click()
  await expect(page.getByRole('button', { name: '正在推理…' })).toBeDisabled()
  await expect(page.getByLabel('待判断文本')).toBeDisabled()
  release()
  await expect(page.getByTestId('answer-value')).toBeVisible()
  expect(requests).toBe(1)
  await expect(page.getByRole('button', { name: '运行推理' })).toBeEnabled()
})

test('校验错误可恢复，服务拒绝保留上次真实结果', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('服务已就绪', { exact: true })).toBeVisible()
  await page.getByLabel('候选项').fill('小李\n小李')
  await page.getByRole('button', { name: '运行推理' }).click()
  await expect(page.getByRole('alert')).toContainText('候选项不能重复')
  await page.getByLabel('快速示例').selectOption('choice')
  await page.getByRole('button', { name: '运行推理' }).click()
  await expect(page.getByTestId('answer-value')).toBeVisible()
  const previous = await page.getByTestId('answer-value').textContent()
  await page.getByLabel('待判断文本').fill('背景材料。'.repeat(2000))
  await page.getByRole('button', { name: '运行推理' }).click()
  await expect(page.getByRole('alert')).toContainText('文本超过模型的 Token 上限')
  await expect(page.getByTestId('answer-value')).toHaveText(previous!)
  await expect(page.getByText('上次成功结果', { exact: true })).toBeVisible()
})

test('窄屏、键盘、长文本与实际对比度', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText('服务已就绪', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: '运行推理' }).click()
  await expect(page.getByTestId('answer-value')).toBeVisible()
  const violations = (await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).analyze()).violations
  expect(violations).toEqual([])
  for (const width of [320, 768, 1280]) {
    await page.setViewportSize({ width, height: 900 })
    await expect(page.getByRole('button', { name: '运行推理' })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    await page.screenshot({ path: `test-results/demo01-${width}.png`, fullPage: true })
  }
  await page.setViewportSize({ width: 1280, height: 900 })
  await page.getByRole('button', { name: '运行推理' }).focus()
  // 真实键盘切换会触发 :focus-visible；程序化 focus 保留鼠标模式。
  await page.keyboard.press('Shift+Tab')
  await page.keyboard.press('Tab')
  await expect(page.getByRole('button', { name: '运行推理' })).toBeFocused()
  const focusStyle = await page.getByRole('button', { name: '运行推理' }).evaluate((element) => getComputedStyle(element).outlineStyle)
  expect(focusStyle).not.toBe('none')
  await page.keyboard.press('Enter')
  await expect(page.getByText('推理完成', { exact: true })).toBeVisible()
  await page.evaluate(() => { document.documentElement.style.zoom = '2' })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: 'test-results/demo01-zoom200.png', fullPage: true })
  await page.evaluate(() => { document.documentElement.style.zoom = '' })
  await page.getByLabel('待判断文本').fill('这是用于检查布局的长文本。'.repeat(100))
  await page.setViewportSize({ width: 320, height: 900 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.emulateMedia({ reducedMotion: 'reduce' })
  const transition = await page.getByRole('button', { name: '运行推理' }).evaluate((element) => getComputedStyle(element).transitionDuration)
  expect(transition).toBe('0s')
})

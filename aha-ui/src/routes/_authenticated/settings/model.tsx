import { createFileRoute } from '@tanstack/react-router'
import { SettingsModel } from '@/features/settings/model'

export const Route = createFileRoute('/_authenticated/settings/model')({
  component: SettingsModel,
})

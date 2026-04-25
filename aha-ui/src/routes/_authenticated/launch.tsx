import { createFileRoute } from '@tanstack/react-router'
import { LaunchPage } from '@/features/launch'

export const Route = createFileRoute('/_authenticated/launch')({
  component: LaunchPage,
})
